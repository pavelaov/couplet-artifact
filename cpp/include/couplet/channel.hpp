// couplet/channel.hpp -- instrumented two-party channels.
//
//   Channel          abstract: send(bytes), recv() -> bytes,
//                    exchange(mine) -> peer (simultaneous-message pattern:
//                    party 0 sends then receives, party 1 receives then
//                    sends; counts one round), flight() (explicit
//                    synchronisation barrier: counts one round), byte /
//                    message counters per direction, and an automatic
//                    "send flight" counter (a maximal run of sends not
//                    interrupted by a receive).
//   TcpChannel       Winsock2 (MSVC, links ws2_32.lib via #pragma) or POSIX
//                    sockets, 4-byte little-endian length framing, robust
//                    send-all / recv-all.  Party 0 listens, party 1 connects
//                    (with retries so start order does not matter).
//   LoopbackChannel  in-process pair over two mutex/condvar queues; usable
//                    from two threads or from a lockstep single-thread
//                    driver (send before recv).
//
// Byte counters count payload bytes only; wire bytes = bytes + 4 * msgs.
//
// Bounded receive.  The length of a received message is chosen by the peer,
// so every receiving call has a limit, and a message above the limit is
// refused with ProtocolAbort:
//
//   recv(max_len, error) / exchange(mine, max_reply, error)
//       the limit is given by the caller, who knows what the protocol allows
//       at this point (the protocol code of this directory passes the exact
//       expected length).  `error` (optional) is the text of the
//       ProtocolAbort, so that a message that is too long is reported like
//       one that is too short;
//   recv() / exchange(mine)
//       the limit is max_message(), a property of the endpoint (default
//       DEFAULT_MAX_MESSAGE = 2^30 bytes; set_max_message changes it).
//
// A transport enforces the limit BEFORE it allocates anything for the
// message: TcpChannel reads the 4-byte length, compares, and only then
// allocates (limit given by the caller) or lets the buffer grow as the bytes
// arrive (endpoint-wide limit), so a 4-byte header cannot make the receiver
// commit more memory than the caller expects or the peer actually sends.
// After a refused frame the byte stream is no longer aligned with the framing
// and the endpoint refuses every further call.  In-process channels hold the
// message already and compare its size.  A refused message is not counted in
// the statistics.  In an exchange, party 1 receives before it sends: when it
// refuses the message of party 0 it has sent nothing of its own half (an
// honest party aborts at the earliest point; honest transcripts are the same
// with and without limits).
//
// struct ProtocolAbort is defined here (it is what a refused message raises);
// hashes.hpp, which includes this header, documents its use.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#if defined(_MSC_VER)
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace couplet {

typedef std::vector<uint8_t> bytes;

// A check on the other party's messages failed (or a message was malformed).
// DESIGN.md change G: the caller must move the whole session to ABORTED.
// This is the only definition of ProtocolAbort in the code base.
#define COUPLET_PROTOCOL_ABORT_DEFINED 1
struct ProtocolAbort : std::runtime_error {
    explicit ProtocolAbort(const std::string& what) : std::runtime_error(what) {}
};

// A received message is longer than the receiving call allows.  Raised by the
// transports; Channel::recv replaces it by a ProtocolAbort with the caller's
// text if one was given.
struct MessageTooLong : ProtocolAbort {
    size_t length, limit;
    MessageTooLong(size_t length_, size_t limit_)
        : ProtocolAbort("received message of " + std::to_string(length_) + " bytes exceeds the limit of " +
                        std::to_string(limit_) + " bytes"),
          length(length_), limit(limit_) {}
};

struct ChannelStats {
    uint64_t bytes_sent = 0;   // payload bytes sent
    uint64_t bytes_recv = 0;   // payload bytes received
    uint64_t msgs_sent = 0;    // framed messages sent
    uint64_t msgs_recv = 0;    // framed messages received
    uint64_t send_flights = 0; // maximal runs of sends (auto-counted)
    uint64_t rounds = 0;       // explicit barriers: flight() and exchange()

    std::string json() const {
        return "{\"bytes_sent\":" + std::to_string(bytes_sent) +
               ",\"bytes_recv\":" + std::to_string(bytes_recv) +
               ",\"msgs_sent\":" + std::to_string(msgs_sent) +
               ",\"msgs_recv\":" + std::to_string(msgs_recv) +
               ",\"send_flights\":" + std::to_string(send_flights) +
               ",\"rounds\":" + std::to_string(rounds) + "}";
    }
};

class Channel {
public:
    // Largest message of the 4-byte length framing used by every transport.
    static constexpr size_t MAX_MESSAGE = 0x7fffffff;
    // Default of max_message(): the limit of a receive whose caller names none.
    static constexpr size_t DEFAULT_MAX_MESSAGE = (size_t)1 << 30;

    explicit Channel(int party) : party_(party), last_(NONE) {
        if (party != 0 && party != 1) throw std::invalid_argument("party must be 0 or 1");
    }
    virtual ~Channel() {}

    int party() const { return party_; }

    // Endpoint-wide limit of recv() / exchange(mine) without an explicit limit.
    size_t max_message() const { return max_message_; }
    void set_max_message(size_t n) {
        if (n > MAX_MESSAGE) throw std::invalid_argument("max_message cannot exceed the frame limit of 2^31 - 1 bytes");
        max_message_ = n;
    }

    void send(const uint8_t* p, size_t n) {
        account_send(n);
        send_raw(p, n);
    }
    void send(const bytes& b) { send(b.empty() ? nullptr : b.data(), b.size()); }
    void send(const std::string& s) { send(s.empty() ? nullptr : (const uint8_t*)s.data(), s.size()); }

    // Receive one message of at most max_message() bytes.
    bytes recv() { return recv_limited(max_message_, false, nullptr); }

    // Receive one message of at most max_len bytes (a limit the caller derives
    // from the protocol; it is not capped by max_message()).  A longer message
    // is refused before it is read: ProtocolAbort with the text `error`, or
    // MessageTooLong if no text is given.
    bytes recv(size_t max_len, const char* error = nullptr) { return recv_limited(max_len, true, error); }

    // Receive and check the exact expected length.
    bytes recv_exact(size_t n) {
        bytes b = recv(n);
        if (b.size() != n) throw std::runtime_error("recv_exact: unexpected message length");
        return b;
    }

    // Simultaneous-message pattern (one round): party 0 sends then receives,
    // party 1 receives then sends.  The reply is limited by max_message().
    bytes exchange(const bytes& mine) { return exchange_limited(mine, max_message_, false, nullptr); }

    // The same with a reply of at most max_reply bytes (see recv(max_len, error)).
    bytes exchange(const bytes& mine, size_t max_reply, const char* error = nullptr) {
        return exchange_limited(mine, max_reply, true, error);
    }

    // Explicit synchronisation barrier: counts one round and starts a new
    // send flight.
    void flight() {
        stats_.rounds++;
        last_ = NONE;
    }

    const ChannelStats& stats() const { return stats_; }
    void reset_stats() { stats_ = ChannelStats(); last_ = NONE; }

protected:
    virtual void send_raw(const uint8_t* p, size_t n) = 0;
    virtual bytes recv_raw() = 0;
    // Receive one message of at most max_len bytes; throws MessageTooLong
    // otherwise.  `from_caller` tells a transport that the limit was named by
    // the caller of the channel (who expects a message of about that size)
    // and is not the endpoint-wide max_message().  A transport that learns
    // the length before the payload overrides this and refuses the message
    // without allocating or reading it; this default suits channels that
    // hold the whole message already.
    virtual bytes recv_raw_bounded(size_t max_len, bool from_caller) {
        (void)from_caller;
        bytes b = recv_raw();
        if (b.size() > max_len) throw MessageTooLong(b.size(), max_len);
        return b;
    }

private:
    enum Last { NONE, SEND, RECV };
    bytes recv_limited(size_t max_len, bool from_caller, const char* error) {
        if (max_len > MAX_MESSAGE) max_len = MAX_MESSAGE;
        bytes b;
        try {
            b = recv_raw_bounded(max_len, from_caller);
        } catch (const MessageTooLong&) {
            if (error) throw ProtocolAbort(error);
            throw;
        }
        account_recv(b.size());
        return b;
    }
    bytes exchange_limited(const bytes& mine, size_t max_reply, bool from_caller, const char* error) {
        bytes peer;
        if (party_ == 0) {
            send(mine);
            peer = recv_limited(max_reply, from_caller, error);
        } else {
            peer = recv_limited(max_reply, from_caller, error);
            send(mine);
        }
        flight();
        return peer;
    }
    void account_send(size_t n) {
        stats_.bytes_sent += n;
        stats_.msgs_sent++;
        if (last_ != SEND) stats_.send_flights++;
        last_ = SEND;
    }
    void account_recv(size_t n) {
        stats_.bytes_recv += n;
        stats_.msgs_recv++;
        last_ = RECV;
    }

    int party_;
    ChannelStats stats_;
    Last last_;
    size_t max_message_ = DEFAULT_MAX_MESSAGE;
};

// ------------------------------------------------------------------ sockets
namespace net {

#if defined(_WIN32)
typedef SOCKET socket_t;
const socket_t invalid_socket = INVALID_SOCKET;
inline void close_socket(socket_t s) { ::closesocket(s); }
inline int last_error() { return WSAGetLastError(); }
inline bool would_retry(int e) { return e == WSAEINTR; }
struct WsaInit {
    WsaInit() {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("WSAStartup failed");
    }
    ~WsaInit() { WSACleanup(); }
};
inline void ensure_init() { static WsaInit init; (void)init; }
#else
typedef int socket_t;
const socket_t invalid_socket = -1;
inline void close_socket(socket_t s) { ::close(s); }
inline int last_error() { return errno; }
inline bool would_retry(int e) { return e == EINTR; }
inline void ensure_init() {}
#endif

inline std::string err_string(const char* what) {
    return std::string(what) + " (error " + std::to_string(last_error()) + ")";
}

inline void set_nodelay(socket_t s) {
    int one = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
}

inline void send_all(socket_t s, const uint8_t* p, size_t n) {
    size_t off = 0;
    while (off < n) {
        size_t chunk = n - off;
        if (chunk > (size_t)0x40000000) chunk = 0x40000000;
#if defined(_WIN32)
        int r = ::send(s, (const char*)p + off, (int)chunk, 0);
        if (r == SOCKET_ERROR) {
            if (would_retry(last_error())) continue;
            throw std::runtime_error(err_string("send failed"));
        }
#else
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        ssize_t r = ::send(s, (const char*)p + off, chunk, flags);
        if (r < 0) {
            if (would_retry(last_error())) continue;
            throw std::runtime_error(err_string("send failed"));
        }
#endif
        off += (size_t)r;
    }
}

inline void recv_all(socket_t s, uint8_t* p, size_t n) {
    size_t off = 0;
    while (off < n) {
        size_t chunk = n - off;
        if (chunk > (size_t)0x40000000) chunk = 0x40000000;
#if defined(_WIN32)
        int r = ::recv(s, (char*)p + off, (int)chunk, 0);
        if (r == SOCKET_ERROR) {
            if (would_retry(last_error())) continue;
            throw std::runtime_error(err_string("recv failed"));
        }
#else
        ssize_t r = ::recv(s, (char*)p + off, chunk, 0);
        if (r < 0) {
            if (would_retry(last_error())) continue;
            throw std::runtime_error(err_string("recv failed"));
        }
#endif
        if (r == 0) throw std::runtime_error("recv: peer closed the connection");
        off += (size_t)r;
    }
}

// The 4-byte little-endian length field of a frame.
inline size_t frame_length(const uint8_t hdr[4]) {
    return (size_t)((uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8) | ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24));
}

// Payload bytes read per step while the buffer of a frame grows (recv_payload).
const size_t RECV_GROW_STEP = (size_t)1 << 24;

// Read the n payload bytes of a frame whose length field has been accepted.
// at_once = true: the caller of the channel expects a message of this size
// and the buffer is allocated in one piece.  Otherwise the length is only
// known to be below the endpoint-wide limit, and the buffer grows in steps of
// RECV_GROW_STEP as the bytes arrive: the peer has to send what it announced
// before the memory is committed.
inline bytes recv_payload(socket_t s, size_t n, bool at_once) {
    bytes b;
    if (at_once || n <= RECV_GROW_STEP) {
        b.resize(n);
        if (n) recv_all(s, b.data(), n);
        return b;
    }
    for (size_t off = 0; off < n;) {
        const size_t step = (n - off < RECV_GROW_STEP) ? (n - off) : RECV_GROW_STEP;
        b.resize(off + step);
        recv_all(s, b.data() + off, step);
        off += step;
    }
    return b;
}

}  // namespace net

class TcpChannel : public Channel {
public:
    static constexpr size_t max_frame = MAX_MESSAGE;

    TcpChannel(int party, net::socket_t fd) : Channel(party), fd_(fd) { net::set_nodelay(fd_); }
    ~TcpChannel() override { close(); }
    TcpChannel(const TcpChannel&) = delete;
    TcpChannel& operator=(const TcpChannel&) = delete;

    void close() {
        if (fd_ != net::invalid_socket) {
            net::close_socket(fd_);
            fd_ = net::invalid_socket;
        }
    }

    // Connect to host:port, retrying until timeout_ms elapses (so the
    // connecting party may start before the listening party).
    static std::unique_ptr<TcpChannel> connect(int party, const std::string& host, uint16_t port,
                                               int timeout_ms = 30000) {
        net::ensure_init();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        std::string last_err;
        for (;;) {
            net::socket_t s = try_connect(host, port, last_err);
            if (s != net::invalid_socket) return std::unique_ptr<TcpChannel>(new TcpChannel(party, s));
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("TcpChannel::connect: " + last_err);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // Party 0 listens on bind_addr:port, party 1 connects to host:port.
    static std::unique_ptr<TcpChannel> create(int party, const std::string& host, uint16_t port,
                                              int timeout_ms = 30000);

protected:
    void send_raw(const uint8_t* p, size_t n) override {
        if (n > max_frame) throw std::runtime_error("TcpChannel: message too large");
        if (desynchronised_) throw std::runtime_error("TcpChannel: the stream is unusable after a refused frame");
        uint8_t hdr[4];
        for (int i = 0; i < 4; i++) hdr[i] = (uint8_t)((uint32_t)n >> (8 * i));
        net::send_all(fd_, hdr, 4);
        if (n) net::send_all(fd_, p, n);
    }
    bytes recv_raw() override { return recv_raw_bounded(max_message(), false); }
    // The length field is compared with the limit before any byte of the
    // payload is read and before any buffer is allocated.
    bytes recv_raw_bounded(size_t max_len, bool from_caller) override {
        if (desynchronised_) throw std::runtime_error("TcpChannel: the stream is unusable after a refused frame");
        uint8_t hdr[4];
        net::recv_all(fd_, hdr, 4);
        const size_t n = net::frame_length(hdr);
        const size_t limit = max_len < (size_t)max_frame ? max_len : (size_t)max_frame;
        if (n > limit) {
            desynchronised_ = true;  // the payload stays unread: the framing is lost
            throw MessageTooLong(n, limit);
        }
        return net::recv_payload(fd_, n, from_caller);
    }

private:
    static net::socket_t try_connect(const std::string& host, uint16_t port, std::string& err) {
        struct addrinfo hints;
        std::memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo* res = nullptr;
        std::string ports = std::to_string(port);
        if (::getaddrinfo(host.c_str(), ports.c_str(), &hints, &res) != 0 || !res) {
            err = "getaddrinfo failed for " + host;
            return net::invalid_socket;
        }
        net::socket_t s = net::invalid_socket;
        for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
            s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (s == net::invalid_socket) continue;
            if (::connect(s, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
            err = net::err_string("connect failed");
            net::close_socket(s);
            s = net::invalid_socket;
        }
        ::freeaddrinfo(res);
        return s;
    }

    net::socket_t fd_;
    bool desynchronised_ = false;
};

class TcpListener {
public:
    // port 0 picks an ephemeral port; see port().
    TcpListener(const std::string& bind_addr, uint16_t port) : fd_(net::invalid_socket), port_(0) {
        net::ensure_init();
        struct addrinfo hints;
        std::memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        struct addrinfo* res = nullptr;
        std::string ports = std::to_string(port);
        if (::getaddrinfo(bind_addr.empty() ? nullptr : bind_addr.c_str(), ports.c_str(), &hints, &res) != 0 || !res)
            throw std::runtime_error("TcpListener: getaddrinfo failed");
        std::string err;
        for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
            net::socket_t s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (s == net::invalid_socket) continue;
#if !defined(_WIN32)
            int one = 1;
            ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
#endif
            if (::bind(s, ai->ai_addr, (int)ai->ai_addrlen) == 0 && ::listen(s, 4) == 0) {
                fd_ = s;
                struct sockaddr_storage ss;
                std::memset(&ss, 0, sizeof ss);
#if defined(_WIN32)
                int len = (int)sizeof ss;
#else
                socklen_t len = (socklen_t)sizeof ss;
#endif
                if (::getsockname(s, (struct sockaddr*)&ss, &len) == 0) {
                    if (ss.ss_family == AF_INET) port_ = ntohs(((struct sockaddr_in*)&ss)->sin_port);
                    else if (ss.ss_family == AF_INET6) port_ = ntohs(((struct sockaddr_in6*)&ss)->sin6_port);
                }
                break;
            }
            err = net::err_string("bind/listen failed");
            net::close_socket(s);
        }
        ::freeaddrinfo(res);
        if (fd_ == net::invalid_socket) throw std::runtime_error("TcpListener: " + err);
    }
    ~TcpListener() {
        if (fd_ != net::invalid_socket) net::close_socket(fd_);
    }
    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;

    uint16_t port() const { return port_; }

    std::unique_ptr<TcpChannel> accept(int party) {
        for (;;) {
            net::socket_t c = ::accept(fd_, nullptr, nullptr);
            if (c == net::invalid_socket) {
                if (net::would_retry(net::last_error())) continue;
                throw std::runtime_error(net::err_string("accept failed"));
            }
            return std::unique_ptr<TcpChannel>(new TcpChannel(party, c));
        }
    }

private:
    net::socket_t fd_;
    uint16_t port_;
};

inline std::unique_ptr<TcpChannel> TcpChannel::create(int party, const std::string& host, uint16_t port,
                                                      int timeout_ms) {
    if (party == 0) {
        TcpListener l(host, port);
        return l.accept(0);
    }
    return TcpChannel::connect(1, host, port, timeout_ms);
}

// ------------------------------------------------------------------ loopback
class LoopbackChannel : public Channel {
public:
    struct Queue {
        std::mutex m;
        std::condition_variable cv;
        std::deque<bytes> q;
        bool closed = false;
    };

    static std::pair<std::unique_ptr<LoopbackChannel>, std::unique_ptr<LoopbackChannel>> make_pair() {
        auto q01 = std::make_shared<Queue>();  // 0 -> 1
        auto q10 = std::make_shared<Queue>();  // 1 -> 0
        std::unique_ptr<LoopbackChannel> c0(new LoopbackChannel(0, q10, q01));
        std::unique_ptr<LoopbackChannel> c1(new LoopbackChannel(1, q01, q10));
        return std::make_pair(std::move(c0), std::move(c1));
    }

    ~LoopbackChannel() override {
        std::lock_guard<std::mutex> lk(out_->m);
        out_->closed = true;
        out_->cv.notify_all();
    }

    // Number of messages waiting to be received (for lockstep drivers).
    size_t pending() const {
        std::lock_guard<std::mutex> lk(in_->m);
        return in_->q.size();
    }

protected:
    void send_raw(const uint8_t* p, size_t n) override {
        bytes b(p, p + n);
        std::lock_guard<std::mutex> lk(out_->m);
        out_->q.push_back(std::move(b));
        out_->cv.notify_one();
    }
    bytes recv_raw() override {
        std::unique_lock<std::mutex> lk(in_->m);
        in_->cv.wait(lk, [&] { return !in_->q.empty() || in_->closed; });
        if (in_->q.empty()) throw std::runtime_error("LoopbackChannel: peer closed");
        bytes b = std::move(in_->q.front());
        in_->q.pop_front();
        return b;
    }

private:
    LoopbackChannel(int party, std::shared_ptr<Queue> in, std::shared_ptr<Queue> out)
        : Channel(party), in_(std::move(in)), out_(std::move(out)) {}

    std::shared_ptr<Queue> in_;
    std::shared_ptr<Queue> out_;
};

}  // namespace couplet
