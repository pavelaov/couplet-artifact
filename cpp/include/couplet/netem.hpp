// couplet/netem.hpp -- application-level network emulation of the native
// two-process drivers (couplet_party, app_e2e, couplet_infer, couplet_item;
// DESIGN.md 9): a one-way latency per message and a token-bucket bandwidth
// limit per direction.
//
//   TokenBucket      a token bucket on the steady clock: rate R bytes per
//                    second, depth C bytes, full at the start.  admit() gives
//                    the time at which the last byte of a frame has its tokens
//                    (fluid model: bytes leave as tokens arrive, so a frame
//                    larger than C is admitted at rate R); admitted_at() gives
//                    the same time for a prefix of the frame.
//   DelayTcpChannel  TCP endpoint with 4-byte length framing that emulates the
//                    latency and the bandwidth limit (this class was part of
//                    cpp/src/couplet_party.cpp; ported from the core's change
//                    P3-BW, see design/CHANGELOG_M3.md entry M3-E; its
//                    behavior without a bandwidth limit is unchanged).
//
// Latency D (--delay-ms):
//
//   * every frame carries the time at which its sender was ready to send it;
//     the receiver hands the message to the protocol no earlier than that
//     time plus D;
//   * a one-way flight therefore costs D;
//   * a simultaneous exchange (Channel::exchange: both parties send, both
//     receive) costs D, not 2 D, although the base class lets party 1 receive
//     before it sends: party 1's part of an exchange is recognized by "a send
//     directly after a receive, with no round boundary in between" and is
//     stamped with the time at which party 1 entered the receive, i.e. the
//     time at which it was ready to send.
//
// Bandwidth limit R (--bandwidth-mbps, in 10^6 bit/s per direction; 0 = off):
//
//   * each endpoint owns the token bucket of its outbound direction (depth C,
//     --bandwidth-bucket-bytes, default DEFAULT_BUCKET_BYTES = one Ethernet
//     frame).  A frame is charged its payload plus the 4-byte length, i.e. the
//     bytes of the plain framing; the 8-byte stamp belongs to the emulation and
//     is not charged, and TCP/IP headers are not modeled;
//   * the bucket admits a frame from the time its sender was ready to send it
//     (the time that the latency emulation stamps, including the rule for the
//     second half of an exchange), and the frame is stamped with the time at
//     which its LAST byte had its tokens.  The receiver hands the message to
//     the protocol no earlier than that time plus D.  On an idle link a frame
//     of m charged bytes thus arrives (m - C) / R + D after its sender was
//     ready (D if m <= C), and a flight of frames after the sum of their
//     transfer times plus D;
//   * the sender is throttled: send() writes the frame in pieces, each piece
//     no earlier than the time at which its last byte had its tokens, and
//     returns only when the whole frame is admitted.  The bucket runs on the
//     emulated clock: the second half of an exchange is admitted from the time
//     its sender entered the receive, so the two directions of an exchange
//     transfer at the same time and do not take turns.  If the emulated
//     transfer of such a half is already over when its sender has received,
//     its bytes are written at once;
//   * one difference from a full-duplex link remains.  The base class lets
//     party 1 receive before it sends, so party 1's half of an exchange can
//     reach party 0 only after party 1 has received party 0's half.  With
//     transfer times T0 (party 0's half) and T1 from equal starts, party 1 is
//     done after max(T0 + D, T1), as on a full-duplex link, and party 0 after
//     max(T0, T1) + D, where a full-duplex link gives max(T0, T1 + D): when
//     party 0's half is the longer one, party 0 waits up to D longer than on
//     the link.  Without a bandwidth limit T0 = T1 = 0 and both are D;
//   * no message is changed: the payload bytes, their order and the channel's
//     counters (payload bytes, messages, rounds) are the same with and without
//     the emulation; only the times at which bytes are written and handed to
//     the protocol differ.
//
// The two processes must run on one machine (the stamps are readings of the
// same monotonic clock).  The 4-byte length and the 8-byte stamp are framing
// and are not counted as payload.  Without a delay and without a bandwidth
// limit no stamp is sent and nothing is throttled: the wire format is then the
// plain 4-byte length framing of TcpChannel (channel.hpp) and of the Python
// package, so that the peer may be another implementation of the protocol
// (tests/interop_peer.py).  Receives are bounded as in TcpChannel: the length
// field is compared with the limit of the receiving call before the payload
// is read or a buffer is allocated.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "channel.hpp"

namespace couplet {
namespace netem {

typedef std::chrono::steady_clock clock;

inline clock::duration from_seconds(double s) {
    return std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(s));
}
inline double to_seconds(clock::duration d) { return std::chrono::duration<double>(d).count(); }

// Bytes per second of a rate in 10^6 bit/s.
inline double mbps_to_bytes_per_second(double mbps) { return mbps * 1e6 / 8.0; }

// Wait until `due`: most of the interval on a timer, the rest spinning.  On Windows the ordinary
// Sleep() has a granularity of up to 15.6 ms, so a high-resolution waitable timer is used (if the
// system has none, the whole interval is spent spinning).  Each thread has its own timer: two
// endpoints in two threads of one process (the tests) must not re-arm each other's timer.
#if defined(_WIN32)
struct WaitTimer {
    HANDLE h;
    WaitTimer() : h(CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS)) {}
    ~WaitTimer() {
        if (h) CloseHandle(h);
    }
    WaitTimer(const WaitTimer&) = delete;
    WaitTimer& operator=(const WaitTimer&) = delete;
};
#endif
inline void coarse_wait(clock::duration d) {
#if defined(_WIN32)
    static thread_local WaitTimer timer;
    if (!timer.h) return;
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() / 100);
    if (due.QuadPart < 0 && SetWaitableTimer(timer.h, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer.h, INFINITE);
#else
    std::this_thread::sleep_for(d);
#endif
}
inline void wait_until(clock::time_point due) {
    for (;;) {
        const clock::time_point now = clock::now();
        if (now >= due) return;
        const clock::duration left = due - now;
        if (left > std::chrono::milliseconds(2)) coarse_wait(left - std::chrono::microseconds(1500));
        else std::this_thread::yield();
    }
}

class TokenBucket {
public:
    // One frame of the bucket's traffic.
    struct Admission {
        clock::time_point start;  // the transfer starts: the sender is ready and the bucket has no backlog
        double tokens = 0;        // tokens in the bucket at `start`, before this frame
        double bytes = 0;         // bytes charged for the frame
        clock::time_point done;   // the last byte has its tokens
    };

    TokenBucket() {}  // inactive: admits everything at once
    TokenBucket(double mbps, double depth_bytes, clock::time_point start) {
        if (!(mbps > 0) || !(mbps <= 1e6)) throw std::invalid_argument("TokenBucket: the rate must be in (0, 10^6] Mbit/s");
        if (!(depth_bytes >= 1) || !(depth_bytes <= 1073741824.0))
            throw std::invalid_argument("TokenBucket: the depth must be in [1, 2^30] bytes");
        rate_ = mbps_to_bytes_per_second(mbps);
        depth_ = depth_bytes;
        tokens_ = depth_bytes;
        t_ = start;
    }

    bool active() const { return rate_ > 0; }
    double rate_bytes_per_second() const { return rate_; }
    double depth_bytes() const { return depth_; }

    // Charge `bytes` for a frame whose sender is ready at `ready`.
    Admission admit(clock::time_point ready, double bytes) {
        Admission a;
        a.start = ready > t_ ? ready : t_;
        a.bytes = bytes;
        if (!active()) {
            a.done = a.start;
            return a;
        }
        a.tokens = std::min(depth_, tokens_ + rate_ * to_seconds(a.start - t_));
        if (bytes <= a.tokens) {
            a.done = a.start;
            tokens_ = a.tokens - bytes;
            t_ = a.start;
        } else {
            a.done = a.start + from_seconds((bytes - a.tokens) / rate_);
            tokens_ = 0;
            t_ = a.done;
        }
        return a;
    }

    // The time at which the first `prefix` charged bytes of the frame `a` have their tokens.
    clock::time_point admitted_at(const Admission& a, double prefix) const {
        if (!active() || prefix <= a.tokens) return a.start;
        if (prefix >= a.bytes) return a.done;
        return a.start + from_seconds((prefix - a.tokens) / rate_);
    }

private:
    double rate_ = 0;     // bytes per second (0: inactive)
    double depth_ = 0;    // bytes
    double tokens_ = 0;   // tokens at t_
    clock::time_point t_; // the bucket's state is known up to here
};

}  // namespace netem

class DelayTcpChannel : public Channel {
public:
    typedef netem::clock clock;

    // Depth of the token bucket unless the caller names one: one Ethernet frame (1,500 bytes).
    static constexpr double DEFAULT_BUCKET_BYTES = 1500.0;
    // Bytes charged per frame besides the payload: the 4-byte length of the plain framing.
    static constexpr double FRAME_CHARGE = 4.0;
    // A throttled frame is written in pieces of the bytes of 4 ms at the rate, at least 64 KiB.
    static constexpr double PIECE_SECONDS = 0.004;
    static constexpr size_t MIN_PIECE = (size_t)1 << 16;

    // bandwidth_mbps = 0: no bandwidth limit (bucket_bytes is then ignored).  Takes ownership of fd
    // (also when an argument is refused).
    DelayTcpChannel(int party, net::socket_t fd, double delay_ms, double bandwidth_mbps = 0.0,
                    double bucket_bytes = DEFAULT_BUCKET_BYTES)
        : Channel(party), fd_(fd) {
        if (!(delay_ms >= 0) || !(delay_ms <= 1e7) || !(bandwidth_mbps >= 0) || !(bandwidth_mbps <= 1e6)) {
            net::close_socket(fd_);
            throw std::invalid_argument("DelayTcpChannel: delay_ms must be in [0, 10^7] and bandwidth_mbps in [0, 10^6]");
        }
        delay_ = std::chrono::nanoseconds((long long)(delay_ms * 1e6));
        if (bandwidth_mbps > 0) {
            try {
                bucket_ = netem::TokenBucket(bandwidth_mbps, bucket_bytes, clock::now());
            } catch (...) {
                net::close_socket(fd_);
                throw;
            }
            piece_ = std::max(MIN_PIECE, (size_t)(bucket_.rate_bytes_per_second() * PIECE_SECONDS));
        }
        bandwidth_mbps_ = bandwidth_mbps;
        net::set_nodelay(fd_);
    }
    ~DelayTcpChannel() override {
        if (fd_ != net::invalid_socket) net::close_socket(fd_);
    }
    DelayTcpChannel(const DelayTcpChannel&) = delete;
    DelayTcpChannel& operator=(const DelayTcpChannel&) = delete;

    // Seconds this endpoint spent waiting for the emulated latency to elapse (receiving side).
    double emulated_wait_s() const { return waited_; }
    // Seconds this endpoint's sends were held back by the token bucket (sending side).
    double bandwidth_wait_s() const { return throttled_; }
    double bandwidth_mbps() const { return bandwidth_mbps_; }
    double bucket_bytes() const { return bucket_.depth_bytes(); }
    double delay_ms() const { return std::chrono::duration<double, std::milli>(delay_).count(); }
    // Whether frames carry the 8-byte stamp (a delay or a bandwidth limit is emulated).
    bool stamped() const { return delay_.count() > 0 || bucket_.active(); }

    static std::unique_ptr<DelayTcpChannel> open(int party, const std::string& host, uint16_t port, double delay_ms,
                                                 int timeout_ms, double bandwidth_mbps = 0.0,
                                                 double bucket_bytes = DEFAULT_BUCKET_BYTES) {
        net::ensure_init();
        return std::unique_ptr<DelayTcpChannel>(new DelayTcpChannel(
            party, party == 0 ? listen_accept(host, port) : connect_retry(host, port, timeout_ms), delay_ms, bandwidth_mbps,
            bucket_bytes));
    }

protected:
    void send_raw(const uint8_t* p, size_t n) override {
        if (n > 0x7fffffffu) throw std::runtime_error("message too large");
        if (desynchronised_) throw std::runtime_error("the stream is unusable after a refused frame");
        // The second half of a simultaneous exchange is stamped with the time this party was ready.
        const clock::time_point ready = (last_recv_ && stats().rounds == rounds_at_recv_) ? recv_entry_ : clock::now();
        last_recv_ = false;
        // With a bandwidth limit the stamp is the time at which the last byte of the frame has its tokens.
        const netem::TokenBucket::Admission adm = bucket_.admit(ready, FRAME_CHARGE + (double)n);
        const clock::time_point stamp = adm.done;
        uint8_t hdr[12];
        for (int i = 0; i < 4; i++) hdr[i] = (uint8_t)((uint32_t)n >> (8 * i));
        const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(stamp.time_since_epoch()).count();
        for (int i = 0; i < 8; i++) hdr[4 + i] = (uint8_t)(ns >> (8 * i));
        if (!bucket_.active() || adm.done <= clock::now()) {
            net::send_all(fd_, hdr, stamped() ? 12 : 4);
            if (n) net::send_all(fd_, p, n);
            return;
        }
        // Throttled: each piece is written no earlier than the time at which its last byte has its tokens.
        hold_until(bucket_.admitted_at(adm, FRAME_CHARGE));
        net::send_all(fd_, hdr, 12);
        for (size_t off = 0; off < n;) {
            const size_t step = std::min(piece_, n - off);
            hold_until(bucket_.admitted_at(adm, FRAME_CHARGE + (double)(off + step)));
            net::send_all(fd_, p + off, step);
            off += step;
        }
        hold_until(adm.done);
    }
    bytes recv_raw() override { return recv_raw_bounded(max_message(), false); }
    bytes recv_raw_bounded(size_t max_len, bool from_caller) override {
        if (desynchronised_) throw std::runtime_error("the stream is unusable after a refused frame");
        const clock::time_point entry = clock::now();
        const uint64_t rounds = stats().rounds;
        uint8_t hdr[12];
        net::recv_all(fd_, hdr, stamped() ? 12 : 4);
        const size_t n = net::frame_length(hdr);
        const size_t limit = max_len < (size_t)MAX_MESSAGE ? max_len : (size_t)MAX_MESSAGE;
        if (n > limit) {
            desynchronised_ = true;  // the payload stays unread: the framing is lost
            throw MessageTooLong(n, limit);
        }
        bytes b = net::recv_payload(fd_, n, from_caller);
        if (stamped()) {
            uint64_t ns = 0;
            for (int i = 7; i >= 0; i--) ns = (ns << 8) | hdr[4 + i];
            const clock::time_point due = clock::time_point(std::chrono::duration_cast<clock::duration>(std::chrono::nanoseconds(ns))) + delay_;
            const clock::time_point before = clock::now();
            netem::wait_until(due);
            const clock::time_point after = clock::now();
            if (after > before) waited_ += std::chrono::duration<double>(after - before).count();
        }
        last_recv_ = true;
        recv_entry_ = entry;
        rounds_at_recv_ = rounds;
        return b;
    }

private:
    void hold_until(clock::time_point t) {
        const clock::time_point before = clock::now();
        if (t <= before) return;
        netem::wait_until(t);
        throttled_ += netem::to_seconds(clock::now() - before);
    }

    static net::socket_t listen_accept(const std::string& host, uint16_t port) {
        struct addrinfo hints;
        std::memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        struct addrinfo* res = nullptr;
        if (::getaddrinfo(host.empty() ? nullptr : host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
            throw std::runtime_error("getaddrinfo failed for " + host);
        net::socket_t ls = net::invalid_socket;
        std::string err = "no usable address";
        for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
            ls = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (ls == net::invalid_socket) continue;
#if !defined(_WIN32)
            int one = 1;
            ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
#endif
            if (::bind(ls, ai->ai_addr, (int)ai->ai_addrlen) == 0 && ::listen(ls, 1) == 0) break;
            err = net::err_string("bind/listen failed");
            net::close_socket(ls);
            ls = net::invalid_socket;
        }
        ::freeaddrinfo(res);
        if (ls == net::invalid_socket) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port) + ": " + err);
        net::socket_t c = net::invalid_socket;
        for (;;) {
            c = ::accept(ls, nullptr, nullptr);
            if (c != net::invalid_socket) break;
            if (!net::would_retry(net::last_error())) {
                net::close_socket(ls);
                throw std::runtime_error(net::err_string("accept failed"));
            }
        }
        net::close_socket(ls);
        return c;
    }
    static net::socket_t connect_retry(const std::string& host, uint16_t port, int timeout_ms) {
        const clock::time_point deadline = clock::now() + std::chrono::milliseconds(timeout_ms);
        std::string err = "no usable address";
        for (;;) {
            struct addrinfo hints;
            std::memset(&hints, 0, sizeof hints);
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            struct addrinfo* res = nullptr;
            if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) == 0 && res) {
                for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
                    const net::socket_t s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                    if (s == net::invalid_socket) continue;
                    if (::connect(s, ai->ai_addr, (int)ai->ai_addrlen) == 0) {
                        ::freeaddrinfo(res);
                        return s;
                    }
                    err = net::err_string("connect failed");
                    net::close_socket(s);
                }
                ::freeaddrinfo(res);
            }
            if (clock::now() >= deadline) throw std::runtime_error("cannot connect to " + host + ":" + std::to_string(port) + ": " + err);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    net::socket_t fd_;
    clock::duration delay_{0};
    netem::TokenBucket bucket_;
    double bandwidth_mbps_ = 0.0;
    size_t piece_ = MIN_PIECE;
    double waited_ = 0.0;
    double throttled_ = 0.0;
    bool desynchronised_ = false;
    bool last_recv_ = false;
    clock::time_point recv_entry_;
    uint64_t rounds_at_recv_ = 0;
};

}  // namespace couplet
