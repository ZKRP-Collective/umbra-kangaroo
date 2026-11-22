// Finite-chain simulator for RCKangaroo-style interval walks.
//
// The group is represented by integer coefficients of a generator.  This is
// exact for collision/recovery while no coefficient wraps the (much larger)
// real curve order.  A deterministic random oracle supplies the x-coordinate
// bits and the orientation bit that real elliptic-curve arithmetic would
// expose.  Consequently x(s) == x(-s), y parity flips under negation, and the
// walk has the same equivalence-class structure without paying for secp256k1
// field arithmetic.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using i64 = std::int64_t;
using u64 = std::uint64_t;

constexpr u64 kOracleSalt = 0xd1b54a32d192ed03ULL;
constexpr u64 kDpSalt = 0x94d049bb133111ebULL;
constexpr u64 kOrientSalt = 0x8538ec25bd456ea3ULL;

u64 mix64(u64 value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

u64 magnitude(i64 value) {
    if (value >= 0) {
        return static_cast<u64>(value);
    }
    // Avoid signed overflow for INT64_MIN.
    return static_cast<u64>(-(value + 1)) + 1;
}

struct Oracle {
    u64 seed;

    u64 x_bits(i64 scalar) const {
        return mix64(magnitude(scalar) ^ seed ^ kOracleSalt);
    }

    bool orientation(i64 scalar) const {
        const bool positive_orientation =
            (mix64(magnitude(scalar) ^ seed ^ kOrientSalt) & 1) != 0;
        return scalar < 0 ? !positive_orientation : positive_orientation;
    }

    bool distinguished(i64 scalar, unsigned bits) const {
        if (bits == 0) {
            return true;
        }
        const u64 mask = (u64{1} << bits) - 1;
        return (mix64(magnitude(scalar) ^ seed ^ kDpSalt) & mask) == 0;
    }
};

struct Rng {
    u64 state;

    u64 next() {
        state += 0x9e3779b97f4a7c15ULL;
        return mix64(state);
    }

    u64 bounded(u64 bound) {
        if (bound == 0) {
            return 0;
        }
        // Lemire-style rejection is unnecessary for these power-of-two-heavy
        // experiments; 128-bit multiply still removes modulo bias.
        return static_cast<u64>(
            (static_cast<unsigned __int128>(next()) * bound) >> 64
        );
    }
};

enum class Policy {
    Baseline,
    PlusParity,
    PlusLite,
    PlusFull,
    PlusFullGated,
    PlusMid2,
    PlusLite3,
    PlusFull3,
    PlusFull4,
    PlusLiteHigh2,
    PlusMidHigh2,
    PlusFullHigh2,
};

std::string policy_name(Policy policy) {
    switch (policy) {
    case Policy::Baseline:
        return "baseline";
    case Policy::PlusParity:
        return "plus-parity";
    case Policy::PlusLite:
        return "plus-lite";
    case Policy::PlusFull:
        return "plus-full";
    case Policy::PlusFullGated:
        return "plus-full-gated";
    case Policy::PlusMid2:
        return "plus-mid2";
    case Policy::PlusLite3:
        return "plus-lite3";
    case Policy::PlusFull3:
        return "plus-full3";
    case Policy::PlusFull4:
        return "plus-full4";
    case Policy::PlusLiteHigh2:
        return "plus-lite-high2";
    case Policy::PlusMidHigh2:
        return "plus-mid-high2";
    case Policy::PlusFullHigh2:
        return "plus-full-high2";
    }
    throw std::logic_error("unknown policy");
}

Policy parse_policy(const std::string& value) {
    if (value == "baseline") {
        return Policy::Baseline;
    }
    if (value == "plus-parity") {
        return Policy::PlusParity;
    }
    if (value == "plus-lite") {
        return Policy::PlusLite;
    }
    if (value == "plus-full") {
        return Policy::PlusFull;
    }
    if (value == "plus-full-gated") {
        return Policy::PlusFullGated;
    }
    if (value == "plus-mid2") {
        return Policy::PlusMid2;
    }
    if (value == "plus-lite3") {
        return Policy::PlusLite3;
    }
    if (value == "plus-full3") {
        return Policy::PlusFull3;
    }
    if (value == "plus-full4") {
        return Policy::PlusFull4;
    }
    if (value == "plus-lite-high2") {
        return Policy::PlusLiteHigh2;
    }
    if (value == "plus-mid-high2") {
        return Policy::PlusMidHigh2;
    }
    if (value == "plus-full-high2") {
        return Policy::PlusFullHigh2;
    }
    throw std::invalid_argument("unknown policy: " + value);
}

struct Config {
    unsigned range_bits = 36;
    unsigned dp_bits = 5;
    unsigned fast_dp_bits = 0;
    unsigned fast_dp_capacity = 0;
    unsigned walkers = 512;
    unsigned jump_count = 512;
    double jump_min_root = 8.0;
    double jump_width_root = 8.0;
    unsigned tame_divisor = 32;
    unsigned edge_divisor = 32;
    double tame_fraction = 1.0 / 3.0;
    unsigned trials = 128;
    unsigned threads = 0;
    unsigned history_length = 10;
    unsigned bins = 0;
    double stage_k = 0.0;
    unsigned stage_tame_divisor = 8;
    double stage_tame_fraction = 0.0;
    double max_k = 12.0;
    u64 seed = 1;
    u64 table_seed = 1;
    u64 oracle_seed = 1;
    bool escape_loops = true;
    bool shadow_dp = false;
    bool csv = false;
    std::vector<Policy> policies{Policy::Baseline};
};

struct Walker {
    i64 scalar = 0;
    i64 distance = 0;
    bool tame = false;
    std::vector<u64> history;
    unsigned history_pos = 0;
    unsigned history_size = 0;
};

struct Record {
    i64 distance;
    bool tame;
};

struct TrialResult {
    u64 secret = 0;
    u64 steps = 0;
    u64 distinguished_points = 0;
    u64 fast_distinguished_points = 0;
    u64 fast_evictions = 0;
    u64 shadow_distinguished_points = 0;
    u64 loop_escapes = 0;
    u64 alternate_evaluations = 0;
    u64 alternate_selections = 0;
    bool solved = false;
};

struct JumpTables {
    std::vector<i64> common;
    std::vector<i64> escape;
};

i64 checked_add(i64 left, i64 right) {
    if ((right > 0 && left > std::numeric_limits<i64>::max() - right) ||
        (right < 0 && left < std::numeric_limits<i64>::min() - right)) {
        throw std::overflow_error("simulated scalar overflow");
    }
    return left + right;
}

JumpTables make_jump_tables(const Config& config) {
    const u64 range = u64{1} << config.range_bits;
    const u64 root = u64{1} << (config.range_bits / 2);
    const u64 common_min = static_cast<u64>(
        std::llround(config.jump_min_root * static_cast<double>(root))
    );
    const u64 common_width = static_cast<u64>(
        std::llround(config.jump_width_root * static_cast<double>(root))
    );
    const u64 escape_min = std::max<u64>(2, range >> 10);
    Rng rng{mix64(config.table_seed ^ 0xa0761d6478bd642fULL)};
    JumpTables tables;
    tables.common.reserve(config.jump_count);
    tables.escape.reserve(config.jump_count);

    for (unsigned i = 0; i < config.jump_count; ++i) {
        u64 jump = common_min + rng.bounded(common_width);
        jump &= ~u64{1};
        tables.common.push_back(static_cast<i64>(jump));
    }
    for (unsigned i = 0; i < config.jump_count; ++i) {
        u64 jump = escape_min + rng.bounded(escape_min);
        jump &= ~u64{1};
        tables.escape.push_back(static_cast<i64>(jump));
    }
    return tables;
}

bool recover_solution(
    const Record& first,
    const Record& second,
    i64 shifted_secret
) {
    if (first.tame && second.tame) {
        return false;
    }

    if (first.tame != second.tame) {
        const i64 tame_distance =
            first.tame ? first.distance : second.distance;
        const i64 wild_distance =
            first.tame ? second.distance : first.distance;
        return wild_distance - tame_distance == shifted_secret ||
               wild_distance + tame_distance == shifted_secret;
    }

    if (first.distance == second.distance) {
        return false;
    }
    const i64 sum = checked_add(first.distance, second.distance);
    return (sum & 1) == 0 && sum / 2 == shifted_secret;
}

bool history_contains(const Walker& walker, u64 point) {
    for (unsigned i = 0; i < walker.history_size; ++i) {
        if (walker.history[i] == point) {
            return true;
        }
    }
    return false;
}

void history_push(Walker& walker, u64 point) {
    if (walker.history.empty()) {
        return;
    }
    walker.history[walker.history_pos] = point;
    walker.history_pos = (walker.history_pos + 1) % walker.history.size();
    walker.history_size =
        std::min<unsigned>(walker.history_size + 1, walker.history.size());
}

struct Candidate {
    i64 scalar;
    i64 delta;
    u64 x;
};

struct Selection {
    Candidate chosen;
    std::optional<Candidate> rejected;
};

Selection select_candidate(
    const Walker& walker,
    const Oracle& oracle,
    i64 jump,
    Policy policy,
    TrialResult& result
) {
    const i64 signed_jump = oracle.orientation(walker.scalar) ? -jump : jump;
    Candidate primary{
        checked_add(walker.scalar, signed_jump),
        signed_jump,
        0,
    };
    primary.x = oracle.x_bits(primary.scalar);

    if (policy == Policy::Baseline) {
        return Selection{primary, std::nullopt};
    }

    unsigned score_bits = 2;
    unsigned score_shift = 0;
    bool evaluate_alternate = false;
    switch (policy) {
    case Policy::Baseline:
        break;
    case Policy::PlusParity:
        score_bits = 1;
        evaluate_alternate = (primary.x & 1) != 0;
        break;
    case Policy::PlusLite:
        evaluate_alternate = (primary.x & 3) == 3;
        break;
    case Policy::PlusFull:
        evaluate_alternate = true;
        break;
    case Policy::PlusFullGated:
        evaluate_alternate = (primary.x & 3) != 0;
        break;
    case Policy::PlusMid2:
        evaluate_alternate = (primary.x & 3) >= 2;
        break;
    case Policy::PlusLite3:
        score_bits = 3;
        evaluate_alternate = (primary.x & 7) == 7;
        break;
    case Policy::PlusFull3:
        score_bits = 3;
        evaluate_alternate = (primary.x & 7) != 0;
        break;
    case Policy::PlusFull4:
        score_bits = 4;
        evaluate_alternate = (primary.x & 15) != 0;
        break;
    case Policy::PlusLiteHigh2:
        score_shift = 32;
        evaluate_alternate = ((primary.x >> score_shift) & 3) == 3;
        break;
    case Policy::PlusMidHigh2:
        score_shift = 32;
        evaluate_alternate = ((primary.x >> score_shift) & 3) >= 2;
        break;
    case Policy::PlusFullHigh2:
        score_shift = 32;
        evaluate_alternate = ((primary.x >> score_shift) & 3) != 0;
        break;
    }
    if (!evaluate_alternate) {
        return Selection{primary, std::nullopt};
    }

    ++result.alternate_evaluations;
    Candidate alternate{
        checked_add(walker.scalar, -signed_jump),
        -signed_jump,
        0,
    };
    alternate.x = oracle.x_bits(alternate.scalar);

    const u64 score_mask = (u64{1} << score_bits) - 1;
    const bool select_alternate =
        ((alternate.x >> score_shift) & score_mask) <
        ((primary.x >> score_shift) & score_mask);

    if (select_alternate) {
        ++result.alternate_selections;
        return Selection{alternate, primary};
    }
    return Selection{primary, alternate};
}

TrialResult run_trial(
    const Config& config,
    const JumpTables& jumps,
    Policy policy,
    unsigned trial_index
) {
    const u64 range = u64{1} << config.range_bits;
    const u64 root = u64{1} << (config.range_bits / 2);
    const u64 tame_range = std::max<u64>(2, range / config.tame_divisor);
    const u64 edge_margin = std::max<u64>(2, range / config.edge_divisor);
    const u64 wild_range = range + 2 * edge_margin;
    const i64 shift = static_cast<i64>(edge_margin);
    const u64 trial_seed =
        mix64(config.seed ^ (u64{trial_index} * 0x9e3779b97f4a7c15ULL));
    Rng rng{trial_seed};
    Oracle oracle{mix64(config.oracle_seed ^ 0xe7037ed1a0b428dbULL)};

    const i64 secret = static_cast<i64>(rng.bounded(range));
    const i64 shifted_secret = checked_add(secret, shift);
    unsigned tame_count = static_cast<unsigned>(
        std::llround(config.walkers * config.tame_fraction)
    );
    tame_count = std::clamp(tame_count, 1U, config.walkers - 1);

    std::vector<Walker> walkers(config.walkers);
    for (unsigned i = 0; i < config.walkers; ++i) {
        Walker& walker = walkers[i];
        walker.tame = i < tame_count;
        u64 distance;
        if (walker.tame) {
            distance = rng.bounded(tame_range);
        } else {
            distance = rng.bounded(wild_range) & ~u64{1};
        }
        walker.distance = static_cast<i64>(distance);
        walker.scalar = walker.tame
            ? walker.distance
            : walker.distance - shifted_secret;
        walker.history.resize(config.history_length);
        history_push(walker, magnitude(walker.scalar));
    }

    std::unordered_map<u64, Record> database;
    database.reserve(static_cast<std::size_t>(root >> config.dp_bits) * 2 + 64);
    std::unordered_map<u64, Record> fast_database;
    std::deque<u64> fast_fifo;
    std::unordered_map<u64, Record> shadow_fast_database;
    std::deque<u64> shadow_fast_fifo;
    if (config.fast_dp_bits != 0) {
        fast_database.reserve(config.fast_dp_capacity * 2 + 64);
        if (config.shadow_dp) {
            shadow_fast_database.reserve(config.fast_dp_capacity * 2 + 64);
        }
    }
    TrialResult result;
    result.secret = static_cast<u64>(secret);
    auto observe_point = [&](
        i64 scalar,
        i64 distance,
        bool tame,
        bool shadow
    ) -> bool {
        const u64 point_class = magnitude(scalar);
        Record current{distance, tame};
        if (config.fast_dp_bits != 0 &&
            oracle.distinguished(scalar, config.fast_dp_bits)) {
            ++result.fast_distinguished_points;
            if (shadow) {
                ++result.shadow_distinguished_points;
            }
            auto collides_with = [&](const auto& table) {
                const auto position = table.find(point_class);
                return position != table.end() &&
                    recover_solution(
                        position->second, current, shifted_secret
                    );
            };
            if (collides_with(fast_database) ||
                collides_with(shadow_fast_database)) {
                return true;
            }

            auto& target_database =
                shadow ? shadow_fast_database : fast_database;
            auto& target_fifo = shadow ? shadow_fast_fifo : fast_fifo;
            const bool inserted =
                target_database.emplace(point_class, current).second;
            if (inserted) {
                target_fifo.push_back(point_class);
                if (target_database.size() > config.fast_dp_capacity) {
                    const u64 evicted = target_fifo.front();
                    target_fifo.pop_front();
                    result.fast_evictions +=
                        target_database.erase(evicted);
                }
            }
        }

        // With a fast tier enabled, shadow samples are intentionally
        // ephemeral. They must not multiply durable HBM usage.
        if (shadow && config.fast_dp_bits != 0) {
            return false;
        }

        if (!oracle.distinguished(scalar, config.dp_bits)) {
            return false;
        }

        ++result.distinguished_points;
        if (shadow && config.fast_dp_bits == 0) {
            ++result.shadow_distinguished_points;
        }
        auto [position, inserted] = database.emplace(point_class, current);
        return !inserted &&
            recover_solution(position->second, current, shifted_secret);
    };
    const u64 max_steps = static_cast<u64>(
        std::ceil(config.max_k * static_cast<double>(root))
    );
    const u64 stage_step = config.stage_k > 0.0
        ? static_cast<u64>(
              std::ceil(config.stage_k * static_cast<double>(root))
          )
        : std::numeric_limits<u64>::max();
    bool stage_applied = false;

    for (; result.steps < max_steps; ++result.steps) {
        if (!stage_applied && result.steps >= stage_step) {
            const u64 stage_tame_range =
                std::max<u64>(2, range / config.stage_tame_divisor);
            if (config.stage_tame_fraction > 0.0) {
                unsigned target_tames = static_cast<unsigned>(
                    std::llround(
                        config.walkers * config.stage_tame_fraction
                    )
                );
                target_tames =
                    std::clamp(target_tames, 1U, config.walkers - 1);
                for (unsigned i = 0; i < walkers.size(); ++i) {
                    Walker& staged = walkers[i];
                    const bool target_tame = i < target_tames;
                    if (staged.tame == target_tame) {
                        continue;
                    }
                    staged.tame = target_tame;
                    u64 distance = target_tame
                        ? rng.bounded(stage_tame_range)
                        : rng.bounded(wild_range) & ~u64{1};
                    staged.distance = static_cast<i64>(distance);
                    staged.scalar = target_tame
                        ? staged.distance
                        : staged.distance - shifted_secret;
                    staged.history_pos = 0;
                    staged.history_size = 0;
                    history_push(staged, magnitude(staged.scalar));
                }
            } else {
                for (Walker& staged : walkers) {
                    if (!staged.tame) {
                        continue;
                    }
                    staged.distance =
                        static_cast<i64>(rng.bounded(stage_tame_range));
                    staged.scalar = staged.distance;
                    staged.history_pos = 0;
                    staged.history_size = 0;
                    history_push(staged, magnitude(staged.scalar));
                }
            }
            stage_applied = true;
        }

        Walker& walker = walkers[result.steps % config.walkers];
        const u64 x = oracle.x_bits(walker.scalar);
        const std::size_t jump_index =
            static_cast<std::size_t>(x % jumps.common.size());
        Selection selection = select_candidate(
            walker, oracle, jumps.common[jump_index], policy, result
        );
        Candidate candidate = selection.chosen;

        if (config.shadow_dp && selection.rejected.has_value()) {
            const Candidate& rejected = *selection.rejected;
            if (observe_point(
                    rejected.scalar,
                    checked_add(walker.distance, rejected.delta),
                    walker.tame,
                    true
                )) {
                ++result.steps;
                result.solved = true;
                return result;
            }
        }

        const u64 candidate_class = magnitude(candidate.scalar);
        if (config.escape_loops &&
            history_contains(walker, candidate_class)) {
            ++result.loop_escapes;
            const i64 escape = jumps.escape[jump_index];
            const i64 signed_escape =
                oracle.orientation(walker.scalar) ? -escape : escape;
            candidate.scalar = checked_add(walker.scalar, signed_escape);
            candidate.delta = signed_escape;
        }

        walker.scalar = candidate.scalar;
        walker.distance = checked_add(walker.distance, candidate.delta);
        const u64 point_class = magnitude(walker.scalar);
        history_push(walker, point_class);

        if (observe_point(
                walker.scalar,
                walker.distance,
                walker.tame,
                false
            )) {
            ++result.steps;
            result.solved = true;
            return result;
        }
    }

    return result;
}

struct Summary {
    double mean_k;
    double stderr_k;
    double median_k;
    double p95_k;
    double p99_k;
    double dp_per_trial;
    double fast_dp_per_trial;
    double fast_evictions_per_trial;
    double shadow_dp_per_trial;
    double escape_rate;
    double alternate_evaluation_rate;
    double alternate_selection_rate;
    unsigned failures;
};

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * (values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Summary summarize(
    const Config& config,
    const std::vector<TrialResult>& results
) {
    const double root = std::pow(2.0, config.range_bits / 2.0);
    std::vector<double> k_values;
    k_values.reserve(results.size());
    u64 total_steps = 0;
    u64 total_dp = 0;
    u64 total_fast_dp = 0;
    u64 total_fast_evictions = 0;
    u64 total_shadow_dp = 0;
    u64 total_escapes = 0;
    u64 total_alt_evaluations = 0;
    u64 total_alt_selections = 0;
    unsigned failures = 0;
    for (const TrialResult& result : results) {
        k_values.push_back(result.steps / root);
        total_steps += result.steps;
        total_dp += result.distinguished_points;
        total_fast_dp += result.fast_distinguished_points;
        total_fast_evictions += result.fast_evictions;
        total_shadow_dp += result.shadow_distinguished_points;
        total_escapes += result.loop_escapes;
        total_alt_evaluations += result.alternate_evaluations;
        total_alt_selections += result.alternate_selections;
        failures += result.solved ? 0 : 1;
    }

    const double mean =
        std::accumulate(k_values.begin(), k_values.end(), 0.0) /
        k_values.size();
    double variance = 0.0;
    for (double value : k_values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= std::max<std::size_t>(1, k_values.size() - 1);

    return Summary{
        mean,
        std::sqrt(variance / k_values.size()),
        percentile(k_values, 0.50),
        percentile(k_values, 0.95),
        percentile(k_values, 0.99),
        static_cast<double>(total_dp) / results.size(),
        static_cast<double>(total_fast_dp) / results.size(),
        static_cast<double>(total_fast_evictions) / results.size(),
        static_cast<double>(total_shadow_dp) / results.size(),
        static_cast<double>(total_escapes) / total_steps,
        static_cast<double>(total_alt_evaluations) / total_steps,
        static_cast<double>(total_alt_selections) / total_steps,
        failures,
    };
}

std::vector<TrialResult> run_trials(
    const Config& config,
    const JumpTables& jumps,
    Policy policy
) {
    std::vector<TrialResult> results(config.trials);
    std::atomic<unsigned> next_trial{0};
    const unsigned hardware_threads =
        std::max(1U, std::thread::hardware_concurrency());
    const unsigned thread_count = config.threads == 0
        ? hardware_threads
        : config.threads;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (unsigned thread_index = 0; thread_index < thread_count; ++thread_index) {
        workers.emplace_back([&] {
            while (true) {
                const unsigned trial = next_trial.fetch_add(1);
                if (trial >= config.trials) {
                    return;
                }
                results[trial] = run_trial(config, jumps, policy, trial);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return results;
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t end = value.find(separator, begin);
        parts.push_back(value.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return parts;
}

u64 parse_u64(const std::string& name, const char* value) {
    std::size_t consumed = 0;
    const std::string text(value);
    const u64 result = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
    return result;
}

double parse_double(const std::string& name, const char* value) {
    std::size_t consumed = 0;
    const std::string text(value);
    const double result = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid " + name + ": " + text);
    }
    return result;
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&]() -> const char* {
            if (++i >= argc) {
                throw std::invalid_argument("missing value after " + arg);
            }
            return argv[i];
        };

        if (arg == "--range-bits") {
            config.range_bits =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--dp-bits") {
            config.dp_bits =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--fast-dp-bits") {
            config.fast_dp_bits =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--fast-dp-capacity") {
            config.fast_dp_capacity =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--walkers") {
            config.walkers =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--jump-count") {
            config.jump_count =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--jump-min-root") {
            config.jump_min_root = parse_double(arg, require_value());
        } else if (arg == "--jump-width-root") {
            config.jump_width_root = parse_double(arg, require_value());
        } else if (arg == "--tame-divisor") {
            config.tame_divisor =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--edge-divisor") {
            config.edge_divisor =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--tame-fraction") {
            config.tame_fraction = parse_double(arg, require_value());
        } else if (arg == "--trials") {
            config.trials =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--threads") {
            config.threads =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--history") {
            config.history_length =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--bins") {
            config.bins =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--stage-k") {
            config.stage_k = parse_double(arg, require_value());
        } else if (arg == "--stage-tame-divisor") {
            config.stage_tame_divisor =
                static_cast<unsigned>(parse_u64(arg, require_value()));
        } else if (arg == "--stage-tame-fraction") {
            config.stage_tame_fraction =
                parse_double(arg, require_value());
        } else if (arg == "--max-k") {
            config.max_k = parse_double(arg, require_value());
        } else if (arg == "--seed") {
            config.seed = parse_u64(arg, require_value());
        } else if (arg == "--table-seed") {
            config.table_seed = parse_u64(arg, require_value());
        } else if (arg == "--oracle-seed") {
            config.oracle_seed = parse_u64(arg, require_value());
        } else if (arg == "--policy") {
            config.policies.clear();
            const std::string value(require_value());
            if (value == "all") {
                config.policies = {
                    Policy::Baseline,
                    Policy::PlusParity,
                    Policy::PlusLite,
                    Policy::PlusFull,
                };
            } else {
                for (const std::string& part : split(value, ',')) {
                    config.policies.push_back(parse_policy(part));
                }
            }
        } else if (arg == "--no-escape") {
            config.escape_loops = false;
        } else if (arg == "--shadow-dp") {
            config.shadow_dp = true;
        } else if (arg == "--csv") {
            config.csv = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: walk_sim [options]\n"
                << "  --range-bits N       interval size 2^N (default 36)\n"
                << "  --dp-bits N          DP probability 2^-N (default 5)\n"
                << "  --fast-dp-bits N     bounded-cache DP predicate\n"
                << "  --fast-dp-capacity N bounded-cache records\n"
                << "  --walkers N          persistent walkers (default 512)\n"
                << "  --jump-count N       jump-table entries (default 512)\n"
                << "  --jump-min-root X    minimum jump / sqrt(N) (default 8)\n"
                << "  --jump-width-root X  random jump width / sqrt(N) (default 8)\n"
                << "  --tame-divisor N     tame width = range/N (default 32)\n"
                << "  --edge-divisor N     edge margin = range/N (default 32)\n"
                << "  --tame-fraction X    walker fraction (default 1/3)\n"
                << "  --trials N           number of secrets (default 128)\n"
                << "  --threads N          worker threads (default hardware)\n"
                << "  --history N          loop history length (default 10)\n"
                << "  --bins N             print K by secret-position bin\n"
                << "  --stage-k X          reseed tames after X*sqrt(N) work\n"
                << "  --stage-tame-divisor N\n"
                << "                       staged tame width (default N/8)\n"
                << "  --stage-tame-fraction X\n"
                << "                       convert walkers to target fraction\n"
                << "  --max-k X            per-trial work cap (default 12)\n"
                << "  --seed N             deterministic seed (default 1)\n"
                << "  --table-seed N       jump-table seed (default 1)\n"
                << "  --oracle-seed N      point-bit oracle seed (default 1)\n"
                << "  --policy LIST        baseline, plus-parity, plus-lite,\n"
                << "                       plus-full, plus-full-gated,\n"
                << "                       plus-mid2, plus-lite3, plus-full3,\n"
                << "                       plus-full4, plus-lite-high2,\n"
                << "                       plus-mid-high2, plus-full-high2,\n"
                << "                       comma list, or all\n"
                << "  --no-escape           disable deterministic loop escape\n"
                << "  --shadow-dp           store rejected SOTA+ candidates\n"
                << "  --csv                 machine-readable summary\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (config.range_bits < 20 || config.range_bits > 48 ||
        (config.range_bits & 1) != 0) {
        throw std::invalid_argument(
            "--range-bits must be even and between 20 and 48"
        );
    }
    if (config.dp_bits > 20) {
        throw std::invalid_argument("--dp-bits must be at most 20");
    }
    if ((config.fast_dp_bits == 0) != (config.fast_dp_capacity == 0)) {
        throw std::invalid_argument(
            "fast DP bits and capacity must be enabled together"
        );
    }
    if (config.fast_dp_bits != 0 &&
        config.fast_dp_bits >= config.dp_bits) {
        throw std::invalid_argument(
            "--fast-dp-bits must be lower than --dp-bits"
        );
    }
    if (config.walkers < 3 || config.trials == 0 || config.jump_count < 2 ||
        config.jump_min_root <= 0.0 || config.jump_width_root <= 0.0 ||
        config.tame_divisor < 2 || config.edge_divisor < 2 ||
        config.stage_tame_divisor < 2 || config.stage_k < 0.0 ||
        config.stage_tame_fraction < 0.0 ||
        config.stage_tame_fraction >= 1.0 ||
        config.max_k <= 0.0) {
        throw std::invalid_argument("invalid non-positive configuration value");
    }
    if (!(config.tame_fraction > 0.0 && config.tame_fraction < 1.0)) {
        throw std::invalid_argument("--tame-fraction must be in (0, 1)");
    }
    return config;
}

void self_test() {
    Oracle oracle{123};
    TrialResult counters;
    Walker positive;
    Walker negative;
    positive.scalar = 987654321;
    negative.scalar = -positive.scalar;
    for (Policy policy : {
             Policy::Baseline,
             Policy::PlusParity,
             Policy::PlusLite,
             Policy::PlusFull,
             Policy::PlusFullGated,
             Policy::PlusMid2,
             Policy::PlusLite3,
             Policy::PlusFull3,
             Policy::PlusFull4,
             Policy::PlusLiteHigh2,
             Policy::PlusMidHigh2,
             Policy::PlusFullHigh2,
         }) {
        TrialResult left_counters;
        TrialResult right_counters;
        const Selection left =
            select_candidate(positive, oracle, 24680, policy, left_counters);
        const Selection right =
            select_candidate(negative, oracle, 24680, policy, right_counters);
        if (left.chosen.scalar != -right.chosen.scalar ||
            left.chosen.delta != -right.chosen.delta) {
            throw std::logic_error("negation-equivariance self-test failed");
        }
    }

    if (!recover_solution(Record{10, true}, Record{90, false}, 100) ||
        !recover_solution(Record{80, false}, Record{120, false}, 100) ||
        recover_solution(Record{10, true}, Record{11, true}, 100)) {
        throw std::logic_error("collision-recovery self-test failed");
    }
    (void)counters;
}

} // namespace

int main(int argc, char** argv) {
    try {
        self_test();
        const Config config = parse_args(argc, argv);
        const JumpTables jumps = make_jump_tables(config);

        if (config.csv) {
            std::cout
                << "policy,range_bits,dp_bits,fast_dp_bits,"
                << "fast_dp_capacity,walkers,jumps,jump_min_root,"
                << "jump_width_root,tame_divisor,"
                << "edge_divisor,tame_fraction,stage_k,"
                << "stage_tame_divisor,stage_tame_fraction,trials,"
                << "seed,table_seed,oracle_seed,failures,mean_k,stderr_k,"
                << "median_k,"
                << "p95_k,p99_k,dp_per_trial,fast_dp_per_trial,"
                << "fast_evictions_per_trial,shadow_dp_per_trial,"
                << "escape_rate,alt_eval_rate,alt_select_rate\n";
        }

        for (Policy policy : config.policies) {
            const std::vector<TrialResult> results =
                run_trials(config, jumps, policy);
            const Summary summary = summarize(config, results);
            if (config.csv) {
                std::cout
                    << policy_name(policy) << ','
                    << config.range_bits << ','
                    << config.dp_bits << ','
                    << config.fast_dp_bits << ','
                    << config.fast_dp_capacity << ','
                    << config.walkers << ','
                    << config.jump_count << ','
                    << config.jump_min_root << ','
                    << config.jump_width_root << ','
                    << config.tame_divisor << ','
                    << config.edge_divisor << ','
                    << config.tame_fraction << ','
                    << config.stage_k << ','
                    << config.stage_tame_divisor << ','
                    << config.stage_tame_fraction << ','
                    << config.trials << ','
                    << config.seed << ','
                    << config.table_seed << ','
                    << config.oracle_seed << ','
                    << summary.failures << ','
                    << summary.mean_k << ','
                    << summary.stderr_k << ','
                    << summary.median_k << ','
                    << summary.p95_k << ','
                    << summary.p99_k << ','
                    << summary.dp_per_trial << ','
                    << summary.fast_dp_per_trial << ','
                    << summary.fast_evictions_per_trial << ','
                    << summary.shadow_dp_per_trial << ','
                    << summary.escape_rate << ','
                    << summary.alternate_evaluation_rate << ','
                    << summary.alternate_selection_rate << '\n';
            } else {
                std::cout
                    << std::fixed << std::setprecision(6)
                    << policy_name(policy)
                    << ": K=" << summary.mean_k
                    << " +/- " << summary.stderr_k
                    << ", median=" << summary.median_k
                    << ", p95=" << summary.p95_k
                    << ", p99=" << summary.p99_k
                    << ", failures=" << summary.failures
                    << ", DP/trial=" << summary.dp_per_trial
                    << ", fast-DP/trial=" << summary.fast_dp_per_trial
                    << ", fast-evict/trial="
                    << summary.fast_evictions_per_trial
                    << ", shadow-DP/trial="
                    << summary.shadow_dp_per_trial
                    << ", escape/step=" << summary.escape_rate
                    << ", alt-eval/step="
                    << summary.alternate_evaluation_rate
                    << ", alt-select/step="
                    << summary.alternate_selection_rate
                    << '\n';
                if (config.bins != 0) {
                    const u64 range = u64{1} << config.range_bits;
                    const double root =
                        std::pow(2.0, config.range_bits / 2.0);
                    std::vector<u64> bin_steps(config.bins);
                    std::vector<unsigned> bin_counts(config.bins);
                    for (const TrialResult& result : results) {
                        const unsigned bin = std::min<unsigned>(
                            config.bins - 1,
                            static_cast<unsigned>(
                                (static_cast<unsigned __int128>(result.secret) *
                                 config.bins) /
                                range
                            )
                        );
                        bin_steps[bin] += result.steps;
                        ++bin_counts[bin];
                    }
                    std::cout << "  bins:";
                    for (unsigned bin = 0; bin < config.bins; ++bin) {
                        const double bin_k = bin_counts[bin] == 0
                            ? 0.0
                            : static_cast<double>(bin_steps[bin]) /
                                bin_counts[bin] / root;
                        std::cout
                            << ' ' << bin << '=' << bin_k
                            << '(' << bin_counts[bin] << ')';
                    }
                    std::cout << '\n';
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "walk_sim: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
