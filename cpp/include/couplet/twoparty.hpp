// couplet/twoparty.hpp -- helpers for running two parties inside one
// process (tests and loopback benchmarks).
//
//   run_two_party(f0, f1)  runs f0(Channel&) and f1(Channel&) on the two
//       endpoints of a LoopbackChannel pair, one std::thread per party, and
//       returns a PartyOutcome per party: the returned value, or the
//       exception text (with a flag telling a ProtocolAbort from any other
//       exception), the endpoint's channel statistics and the wall-clock
//       seconds of the party function.  When a party function returns or
//       throws, its endpoint is destroyed, so a peer that is still waiting
//       for a message is woken up with an exception instead of blocking
//       forever.  The return types must be default-constructible.
//
//   TamperChannel  a channel proxy for adversarial tests: it rewrites
//       selected outgoing messages of a party (rule k applies to the k-th
//       message the endpoint sends, counting from 0) and forwards everything
//       else unchanged.
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "channel.hpp"
#include "hashes.hpp"

namespace couplet {

template <class R>
struct PartyOutcome {
    bool ok = false;              // the party function returned normally
    bool protocol_abort = false;  // it threw ProtocolAbort
    std::string error;            // what() of the exception, if any
    R value{};                    // the returned value (valid if ok)
    ChannelStats stats;           // statistics of the party's endpoint
    double seconds = 0.0;         // wall-clock time of the party function

    // True iff the party ended with an exception of any kind.
    bool failed() const { return !ok; }
};

namespace detail {
template <class F, class R>
inline void run_party(F& fn, std::unique_ptr<LoopbackChannel> ch, PartyOutcome<R>& out) {
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    try {
        out.value = fn(static_cast<Channel&>(*ch));
        out.ok = true;
    } catch (const ProtocolAbort& e) {
        out.protocol_abort = true;
        out.error = e.what();
    } catch (const std::exception& e) {
        out.error = e.what();
    } catch (...) {
        out.error = "unknown exception";
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out.stats = ch->stats();
    ch.reset();  // closes the outgoing queue: a waiting peer gets an exception
}
}  // namespace detail

template <class F0, class F1>
inline auto run_two_party(F0&& f0, F1&& f1)
    -> std::pair<PartyOutcome<typename std::decay<decltype(f0(std::declval<Channel&>()))>::type>,
                 PartyOutcome<typename std::decay<decltype(f1(std::declval<Channel&>()))>::type>> {
    typedef typename std::decay<decltype(f0(std::declval<Channel&>()))>::type R0;
    typedef typename std::decay<decltype(f1(std::declval<Channel&>()))>::type R1;
    std::pair<PartyOutcome<R0>, PartyOutcome<R1>> out;
    std::pair<std::unique_ptr<LoopbackChannel>, std::unique_ptr<LoopbackChannel>> chans = LoopbackChannel::make_pair();
    std::unique_ptr<LoopbackChannel> c0 = std::move(chans.first), c1 = std::move(chans.second);
    std::thread t1([&f1, &out, &c1] { detail::run_party(f1, std::move(c1), out.second); });
    detail::run_party(f0, std::move(c0), out.first);
    t1.join();
    return out;
}

// Channel proxy that rewrites selected outgoing messages (tests).
class TamperChannel : public Channel {
public:
    typedef std::function<bytes(const bytes&)> Rule;

    TamperChannel(Channel& inner, std::map<size_t, Rule> rules)
        : Channel(inner.party()), inner_(&inner), rules_(std::move(rules)), sent_(0) {}

    size_t messages_sent() const { return sent_; }

protected:
    void send_raw(const uint8_t* p, size_t n) override {
        bytes msg(p, p + n);
        const std::map<size_t, Rule>::const_iterator it = rules_.find(sent_++);
        if (it != rules_.end()) msg = it->second(msg);
        inner_->send(msg);
    }
    bytes recv_raw() override { return inner_->recv(); }
    // The limit of the receiving call is handed to the wrapped endpoint.
    bytes recv_raw_bounded(size_t max_len, bool from_caller) override {
        bytes b = from_caller ? inner_->recv(max_len) : inner_->recv();
        if (b.size() > max_len) throw MessageTooLong(b.size(), max_len);
        return b;
    }

private:
    Channel* inner_;
    std::map<size_t, Rule> rules_;
    size_t sent_;
};

// Message rewriters for TamperChannel.
inline TamperChannel::Rule flip_bit_rule(size_t position) {
    return [position](const bytes& in) {
        bytes out = in;
        if (position / 8 < out.size()) out[position / 8] ^= (uint8_t)(1u << (position % 8));
        return out;
    };
}
inline TamperChannel::Rule truncate_rule(size_t drop) {
    return [drop](const bytes& in) { return bytes(in.begin(), in.end() - (ptrdiff_t)(drop < in.size() ? drop : in.size())); };
}
inline TamperChannel::Rule append_rule(size_t extra) {
    return [extra](const bytes& in) {
        bytes out = in;
        out.resize(out.size() + extra, 0);
        return out;
    };
}

}  // namespace couplet
