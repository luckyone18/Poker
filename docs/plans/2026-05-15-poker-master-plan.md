# Poker AI Bot — Master Plan

> **Goal:** Build a poker AI that wins consistently on crypto poker sites, starting with SWC Poker (browser-based Bitcoin NLHE).
> **Repo:** [luckyone18/Poker](https://github.com/luckyone18/Poker) — based on thotbreakerr/Texas-Holdem-AI

---

## Project Structure

```
~/Poker/
├── core/              # NLHE engine: betting rounds, hand evaluator, side pots
├── bots/              # 9 AI bot implementations (Monte Carlo, PPO RL, CFR, GTO, ICM, etc.)
├── training/          # 4 training pipelines (self-play RL, multi-agent PPO, CFR, supervised ML)
├── models/            # Saved weights + five_card_table.pkl (~45MB, auto-generated)
├── run_local_match.py      # Single tournament runner → output/ chart
├── run_tournament.py       # Live tournament UI (matplotlib)
├── run_tournament_stats.py # Batch statistics (multiprocessing)
└── docs/plans/        # Implementation plans
```

---

## Phase 1: Foundation — Training a Winning Bot

**Duration: 2-3 weeks** (running on Vultr server, CPU-only PyTorch 2.12)

### Milestone 1.1 — Fix Upstream Bugs & Baseline

| Task | Description | Priority |
|------|-------------|----------|
| 1.1.1 | Fix SmartBot crash (`_estimate_equity` — empty `opp_hands`) | 🔴 High |
| 1.1.2 | Run baseline benchmark: 500 tournaments MC200 vs CFR vs GTO vs ICM, establish win rates | 🔴 High |
| 1.1.3 | Fix BB position-encoding feature mismatch (`ml_bot.py` 0.3 vs `train_ml_bot.py` 0.5) | 🔴 High |
| 1.1.4 | Drop fixed 10% exploration in RL bot — use PPO entropy bonus instead | 🟡 Medium |

**Verification:** All 9 bots run without crashes across 500 tournaments. Baseline win rates documented.

### Milestone 1.2 — Generate Training Dataset

| Task | Description | Priority |
|------|-------------|----------|
| 1.2.1 | Run `run_tournament_stats.py` with strong bots only (cfr, mc200, gto, exploitative, icm), 6-player tables, 500 tournaments | 🔴 High |
| 1.2.2 | Enable JSONL decision logging (`core/logger.py`) to capture ~100k decisions | 🔴 High |
| 1.2.3 | Vary blind levels and stack depths: early-game (deep stacks), mid-game, late-game (short-stack, bubble, heads-up) | 🔴 High |
| 1.2.4 | Retrain CFR for 6-player deep-stack (`train_cfr_bot_multiway.py`), overnight background run | 🟡 Medium |

**Verification:** `logs/` directory populated with JSONL decision logs from all tournament stages.

### Milestone 1.3 — Supervised Warm-Start

| Task | Description | Priority |
|------|-------------|----------|
| 1.3.1 | Train ML bot via supervised learning on CFR decisions (`train_ml_bot.py --filter_players P_cfr`) | 🔴 High |
| 1.3.2 | Copy trained ML weights into RL bot's policy network as initialization | 🔴 High |
| 1.3.3 | Validate: warm-started bot should roughly break even vs CFR heads-up in 100 matches | 🔴 High |
| 1.3.4 | Train ML bot on ALL strong bots (not just CFR) for broader coverage | 🟡 Medium |

**Verification:** Warm-started RL bot achieves 45%+ win rate vs CFR in 100 heads-up matches.

### Milestone 1.4 — Self-Play RL Training

| Task | Description | Priority |
|------|-------------|----------|
| 1.4.1 | Run `train_rl_bot_selfplay.py` with warm-started weights (skip random/heuristic curriculum stages) | 🔴 High |
| 1.4.2 | Target: 20k-50k episodes | 🔴 High |
| 1.4.3 | Monitor: reward/hand increasing, entropy not collapsing, win rate improving | 🔴 High |
| 1.4.4 | Save checkpoint every 500 episodes to `models/rl_selfplay_snapshot.pt` | 🔴 High |
| 1.4.5 | After self-play: run `train_multi_deep_rl_bot.py` vs CFR + MC200 + GTO | 🟡 Medium |

**Verification:** Trained RL bot wins 25%+ in 6-player tournaments (random baseline = ~14%).

### Milestone 1.5 — League Training & Evaluation

| Task | Description | Priority |
|------|-------------|----------|
| 1.5.1 | Implement league pool with rotating opponents: recent snapshots + CFR + MC200 + GTO + heuristic | 🟡 Medium |
| 1.5.2 | Run 500-1000 tournament eval with Wilson confidence intervals | 🟡 Medium |
| 1.5.3 | Target: bot achieves 30%+ tournament win rate against mixed field | 🟡 Medium |

**Verification:** Bot wins at least 2x random baseline in 6-player tournaments.

### Milestone 1.6 — Architecture Upgrades (If time permits)

| Task | Description | Priority |
|------|-------------|----------|
| 1.6.1 | Expand features: card embeddings, per-street opponent stats | 🟢 Nice-to-have |
| 1.6.2 | Add betting history encoder (GRU) | 🟢 Nice-to-have |
| 1.6.3 | Implement AIVAT-style equity reward instead of chip delta | 🟡 Medium |
| 1.6.4 | Continuous bet sizing (9-12 discrete buckets) | 🟢 Nice-to-have |

---

## Phase 2: Real-Money Integration — SWC Poker

**Duration: 2-4 weeks** (after Phase 1 bot is winning consistently)

### Pre-requisite Research

Before writing detailed Phase 2 plan,我们需要research hal-hal berikut:

1. **SWC Poker architecture audit**
   - API endpoints (WebSocket? REST? Socket.IO?)
   - Authentication flow (Bitcoin deposit/withdraw, session tokens)
   - Game state protocol (how does the client receive cards, actions, pot updates)
   - Anti-bot measures (CAPTCHA? timing checks? behavior analysis?)

2. **Legal & Risk assessment**
   - SWC Poker ToS — botting policy
   - Bitcoin wallet management (hot wallet for buy-ins vs cold storage)
   - Bankroll management strategy (Kelly criterion for poker)

### Milestone 2.1 — SWC Poker Reverse Engineering

| Task | Description | Priority |
|------|-------------|----------|
| 2.1.1 | Create SWC Poker account, deposit minimum Bitcoin | 🔴 High |
| 2.1.2 | Capture browser network traffic (Chrome DevTools + mitmproxy) during a full session | 🔴 High |
| 2.1.3 | Document: login flow, lobby/table join, hand lifecycle, cash-out flow | 🔴 High |
| 2.1.4 | Identify: WebSocket endpoints, message formats, auth tokens | 🔴 High |
| 2.1.5 | Write standalone Python client that connects to SWC and receives game state | 🔴 High |

### Milestone 2.2 — Bridge: Engine → SWC Client

| Task | Description | Priority |
|------|-------------|----------|
| 2.2.1 | Map SWC game state → `PlayerView` (our engine's interface) | 🔴 High |
| 2.2.2 | Map our `Action` → SWC action messages | 🔴 High |
| 2.2.3 | Handle: multi-tabling support (play multiple tables simultaneously) | 🟡 Medium |
| 2.2.4 | Handle: disconnection recovery, table re-join | 🟡 Medium |

### Milestone 2.3 — Bot Safety & Anti-Detection

| Task | Description | Priority |
|------|-------------|----------|
| 2.3.1 | Add randomized action delays (human-like timing distribution: 2-8s per decision) | 🔴 High |
| 2.3.2 | Add "tells" simulation: occasional suboptimal plays, varied bet sizing | 🟡 Medium |
| 2.3.3 | Session management: auto-login, table selection, stop-loss limits | 🟡 Medium |

### Milestone 2.4 — Live Testing & Bankroll Management

| Task | Description | Priority |
|------|-------------|----------|
| 2.4.1 | Start with micro-stakes tables (lowest buy-in) | 🔴 High |
| 2.4.2 | Track: bb/100 hands, ROI, hourly rate | 🔴 High |
| 2.4.3 | Implement automatic stop-loss (quit after losing X buy-ins) | 🟡 Medium |
| 2.4.4 | After 1 week profitable at micros → move up stakes | 🟢 Nice-to-have |

---

## Phase 3: Expansion (Optional/Future)

### Other Crypto Poker Sites

| Site | Viability | Notes |
|------|-----------|-------|
| SWC Poker | ✅ Primary target | Browser-based, Bitcoin, simpler protocol |
| CoinPoker | ⚠️ Harder | App-based (Windows/Mac), anti-bot detection |
| Americas Cardroom | ⚠️ Harder | App-based, aggressive anti-bot |
| BetOnline/Ignition | 🔮 Unknown | Browser-based but complex CAPTCHA |

### Training Improvements

- **Deep CFR / ReBeL**: Replace PPO with purpose-built imperfect-information game algorithms
- **Test-time search**: DeepStack-style shallow lookahead at decision time
- **Multi-table RL**: Train bot to play 4+ tables simultaneously
- **Player profiling**: Build opponent models from observed play, select counter-strategies

---

## Infrastructure & DevOps

### Current Setup
- **Server:** Vultr VM (SG region), CPU-only, Python 3.11 venv
- **Repo:** [luckyone18/Poker](https://github.com/luckyone18/Poker)
- **Dependencies:** PyTorch 2.12, matplotlib, treys

### Server Limitations
- **IP range blocked** by most gambling sites (datacenter IP)
- **No GPU** → RL training is CPU-bound (~10x slower than M5 Max)
- **For SWC live play:**可能需要proxy/residential IP or browser-based approach

### Workarounds for SWC
- Option A: Run browser + bot on local machine (not server), use server for training only
- Option B: Residential proxy + headless browser on server (expensive, detectable)
- Option C: Local machine runs SWC client, server runs AI decision engine (API bridge)

---

## Success Metrics

| Metric | Phase 1 Target | Phase 2 Target |
|--------|---------------|----------------|
| Tournament win rate (6-player) | 30%+ (vs 14% random) | N/A |
| Heads-up win rate vs CFR | 50%+ | N/A |
| bb/100 hands (live play) | N/A | 5+ bb/100 at micros |
| Monthly profit (after 3 months) | N/A | $200+ at micros |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-only training too slow | Schedule slip | Run overnight, prioritize CFR (fastest to train) |
| SWC patches bot detection | Bot unusable | Research multiple sites, maintain fallback options |
| Server IP blocked by SWC | Cannot connect | Run client on local machine with residential IP |
| Bot loses money live | Financial loss | Start micro-stakes, strict stop-loss, paper-trade first |
| SWC ToS violation → funds confiscated | Total loss | Never deposit more than willing to lose, withdraw frequently |

---

## What We Already Have ✅

- ✅ Full NLHE engine with hand evaluator (2.6M 5-card combos)
- ✅ 9 bot AI implementations (MC, RL, CFR, GTO, ICM, exploitative, opponent model, heuristic, ML)
- ✅ 4 training pipelines (self-play RL, multi-agent PPO, CFR, supervised ML)
- ✅ Tournament runner with chip history visualization
- ✅ Repo live at [luckyone18/Poker](https://github.com/luckyone18/Poker)
- ✅ Verified: engines runs, tournaments complete, sanity tests pass

---

## Immediate Next Actions

1. **Fix SmartBot crash** — 1-line fix in `poker_mind_bot.py:_estimate_equity`
2. **Generate dataset** — run 500 tournaments with logging enabled
3. **Retrain CFR overnight** — `train_cfr_bot_multiway.py` background run
4. **Start supervised warm-start** — train ML on CFR decisions, port to RL