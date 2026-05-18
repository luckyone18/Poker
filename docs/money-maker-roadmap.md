# Poker Bot — Money Maker Roadmap

## Context
**Goal:** Generate consistent profit from crypto poker (SWC Poker / swcpoker.com)
**Stack:** Texas-Holdem-AI engine + custom RL training + SWC WebSocket integration
**Status:** Engine & basic bots exist. RL training broken (1% WR vs SmartBot). SWC parser is skeleton only.

---

## Core Insight
A poker bot earns money through **volume × edge**. Even a simple bot with 1-3bb/100 win rate printing money at 10+ tables. Priority:

1. **Get AI working in simulation** (prove RL pipeline works)
2. **Deploy to real platform** (SWC Poker WebSocket integration)
3. **Operate profitably** (bankroll management, anti-detection)

---

## PHASE 1 — RL Training Pipeline (Week 1-2)
**Goal: Proven RL pipeline with >55% WR vs SmartBot in simulation**

### 1.1 Fix RL Training Bug
**Problem:** BC policy is 68.5% vs RandomBot but only 1% vs SmartBot. After 10K RL training, drops to 0.5% vs SmartBot.
**Root cause:** Reward signal degenerate — `record_reward()` called AFTER hand ends, but PPO filters steps by `'reward' in step` which only works for steps WITH reward keys. The `record_reward()` injects reward into ALL steps of the episode, but this happens AFTER the episode buffer was already flushed.

**Fix needed:**
- Call `rl_bot.record_reward(reward)` BEFORE `rl_bot.end_episode()` so the reward injection happens while the episode is still open
- Verify `_ppo_update()` actually processes episodes (check if steps have `'reward'` key)
- Benchmark BEFORE and AFTER fix

**Deliverable:** RL training that produces measurable improvement over BC baseline

### 1.2 Proper Training Setup
- **Opponent:** SmartBot (competent, ~50% vs itself = good baseline)
- **Position randomization:** Alternate or randomize seat positions so RL learns both BB and SB
- **Stakes:** Start at 2000 chips (100bb), maybe test 500 chips for faster games
- **Reward:** Normalized chip delta (current - starting) / starting
- **Metrics to track:** Win rate, avg chips, hands/hour, showdown rate

**Verification:** At minimum, RL model must beat BC policy against SmartBot

### 1.3 Train & Benchmark
- 20K episodes training on Modal T4
- Save checkpoints every 5K
- Benchmark each checkpoint: vs RandomBot, vs SmartBot, vs MC(200)
- Pick best model

**Target:** >55% WR vs SmartBot at 2000 chips, 200 episodes, no exploration

---

## PHASE 2 — AI Engine (Week 2-3)
**Goal: Reliable +EV decision engine for real-money play**

### 2.1 Hybrid AI Strategy
Combine multiple approaches:
- **Primary:** RL-trained policy network (fast inference, ~1ms/decision)
- **Fallback:** GTO approximations or rule-based for edge cases
- **Safety:** Auto-fold if decision time >5s (network/platform timeout)

### 2.2 Opponent Modeling
- Track opponent tendencies (VPIP, PFR, fold frequency) per session
- Adapt exploitative adjustments on the fly
- SWC Poker players are generally recreational = exploitable

### 2.3 Multi-Street Decision Quality
- Benchmark each street (preflop, flop, turn, river) separately
- Identify weakest street and fix
- Ensure showdown value extraction (don't fold winning hands)

**Deliverable:** AI that makes decisions in <100ms, wins consistently vs recreational players

---

## PHASE 3 — SWC Poker Integration (Week 3-5)
**Goal: Bot plays real money on SWC Poker**

### 3.1 SWC Reverse Engineering
**What we know:** SWC Poker is browser-based (HTML5), Bitcoin-funded
**What we need:** WebSocket protocol, authentication, game state, action submission

**Steps:**
1. Capture SWC Poker network traffic (Chrome DevTools)
2. Parse WebSocket messages (swc_parser.py skeleton exists)
3. Map SWC events → internal PlayerView format
4. Handle authentication (session cookies / tokens)
5. Test with small deposit (0.001 BTC)

**Known challenges:**
- Server IP (108.61.187.219) is datacenter IP — likely flagged by poker sites
- Need to appear as residential user or use proxy
- SWC may have own bot detection

### 3.2 Deposit & Withdrawal Flow
- Bitcoin deposit to SWC Poker
- Convert BTC → play chips
- Withdraw back to BTC wallet
- Track ROI per session

### 3.3 Integration with Poker Engine
- Bridge: SWC events → internal game state → RLBot decision → SWC action
- Latency: Must act before SWC timeout (typically 15-30s)
- Reconnection: Handle disconnects gracefully

**Deliverable:** Bot that can join table, play hands, and cash out on SWC Poker

---

## PHASE 4 — Live Operations (Ongoing)
**Goal: Generate consistent profit**

### 4.1 Bankroll Management
- **Kelly Criterion:** Risk max 1-2% of bankroll per session
- **Session stops:** Stop loss (-2 buyins) / Stop win (+3 buyins)
- **Variance buffer:** 20+ buyins minimum before going "all in"

### 4.2 Table Selection
- Prefer: Loose tables, recreational players, higher stakes (more fish)
- Avoid: Tight tables, other bots, professional players
- Multi-table if bot is profitable

### 4.3 Anti-Detection
- Randomized timing (not robotic precision)
- Occasional "human-like" folds/calls (bluff failures)
- Vary bet sizing slightly
- Don't play too many hands per hour (humans play ~60 hands/hr)
- Use residential proxy, not datacenter IP

### 4.4 Monitoring & Alerts
- Track win rate, hours played, ROI per session
- Alert on unusual patterns (winning too fast = might be flagged)
- Withdraw profits regularly

**Financial target:** $10-50/day profit at launch → scale as bot improves

---

## Critical Risks
| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| SWC detects & bans bot | High | Loses ability to earn | Residential proxy, timing randomization |
| Bot loses money (negative EV) | Medium | Financial loss | Strict bankroll management, start micro |
| SWC Poker shuts down / scam | Low-Medium | Lose bankroll | Only play on established platform, withdraw frequently |
| Bitcoin volatility | Medium | USD value fluctuates | Withdraw to stablecoin or cash out immediately |

---

## Success Metrics
| Milestone | Target | Timeline |
|---|---|---|
| RL pipeline fixed | >55% WR vs SmartBot | Week 2 |
| Hybrid AI trained | Beats SmartBot + exploits RandomBot | Week 3 |
| First real hand on SWC | Bot joins table + plays | Week 4 |
| Break even (recover deposit) | Net profit > $0 | Week 5 |
| Consistent profit | $10+/day average | Week 6+ |

---

## Current State
- [x] Poker engine (core/engine.py)
- [x] RLBot, SmartBot, MC Bot, GTO Bot
- [x] BC policy trained (68.5% vs RandomBot, 1% vs SmartBot)
- [x] Modal training setup (T4 GPU)
- [ ] RL training pipeline (BROKEN - 1% WR)
- [ ] SWC WebSocket integration (parser skeleton only)
- [ ] Live bot on SWC Poker
- [ ] Profitability tracking

## Next Action
**Start with Phase 1.1: Fix the RL training bug.**
See `docs/rl-training-debug-log.md` for detailed bug analysis.