# AI Driver 🚗

AI Driver is an open-source, educational research project focused on **learning autonomous driving behaviors through reinforcement learning** using lightweight simulation environments such as `highway-env`.

Rather than aiming to build a full self-driving system, the project explores **how driving capabilities can be progressively learned, evaluated, and composed**, starting from basic parking maneuvers and advancing toward point‑to‑point navigation.

---

## 📋 Table of Contents

* [Overview](#overview)
* [Learning Approach](#learning-approach)
* [Project Goals](#project-goals)
* [Roadmap](#roadmap)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)
* [Documentation](#documentation)

---

## 🎯 Overview

AI Driver explores autonomous driving as a **progressive learning problem**.

Instead of end‑to‑end autonomy, the project is structured as a sequence of increasingly complex scenarios:

* Goal‑conditioned parking
* Parking with obstacles
* Controlled road navigation
* Traffic‑rule compliance
* Scenario‑based driving (roundabouts, highways)
* High‑level navigation from Point A to Point B

Each stage is intentionally **small, interpretable, and reproducible**, allowing close inspection of learning dynamics and failure modes.

The entire journey is documented as a **public LinkedIn series**, highlighting design decisions, trade‑offs, and lessons learned while building autonomous driving agents.

---

## 🧠 Learning Approach

AI Driver is built using **reinforcement learning (RL)**, where an agent learns driving behaviors through interaction with a simulated environment.

The project follows a layered autonomy stack:

1. **Low‑level control** – steering, throttle, braking
2. **Goal‑conditioned tasks** – e.g. parking in a target pose
3. **Scenario learning** – intersections, roundabouts, highways
4. **High‑level navigation** – route planning and execution

Early stages focus on **continuous control** and **goal‑conditioned RL**, using algorithms such as:

* SAC (Soft Actor‑Critic)
* PPO (Proximal Policy Optimization)
* HER (Hindsight Experience Replay) for sparse‑reward tasks

The emphasis is on **understanding how and why agents learn**, not only on final performance.

---

## 🎯 Project Goals

The long‑term objective is to explore how autonomous driving capabilities can be progressively learned and composed, including:

* Reaching a target destination on a map
* Executing parking maneuvers
* Following lanes and controlling speed
* Respecting basic traffic rules
* Handling structured road scenarios
* Making safe, smooth, and efficient driving decisions

This project prioritizes **clarity, realism, and learning value** over completeness.

---

## 🗺️ Roadmap

Each phase is intentionally scoped to remain lightweight and focused.

### Phase 1: Foundation ✅ (In Progress)

* [x] Project setup
* [x] Environment configuration
* [x] Parking environment baseline
* [x] Random agent benchmark

### Phase 2: Parking & Obstacles

* [x] Goal‑conditioned parking (empty lot)
    ```bash
    python src/evaluation/evaluate_parking.py --model-path models/parking/sac_her_20251216_222821/best/best_model.zip --render --episodes 10
    ```
* [ ] Parking with static obstacles
* [ ] Parking with constrained space
* [ ] Evaluation metrics and success rate

### Phase 3: Road Navigation

* [ ] Basic road following
* [ ] Lane keeping
* [ ] Speed control
* [ ] Simple turns

### Phase 4: Traffic Rules

* [ ] Stop signs
* [ ] Yield behavior
* [ ] Traffic lights
* [ ] Right‑of‑way logic

### Phase 5: Complex Scenarios

* [ ] Roundabouts
* [ ] Highway merging
* [ ] Lane changes
* [ ] Overtaking

### Phase 6: Point‑to‑Point Navigation

* [ ] Map generation or loading
* [ ] High‑level path planning
* [ ] Route following
* [ ] End‑to‑end navigation experiments

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* pip

### Setup

```bash
git clone https://github.com/yourusername/AiDriver.git
cd AiDriver
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Sanity Check

Verify that the environment runs correctly:

```bash
python sanity_run.py
```

This launches a parking environment with a random agent.

### Random Agent Benchmark

Run a baseline benchmark with a random agent to establish performance metrics:

```bash
python run_benchmark.py --episodes 100
```

Options:
- `--episodes N`: Number of episodes to run (default: 100)
- `--render`: Render the environment during evaluation
- `--seed N`: Set random seed for reproducibility
- `--output PATH`: Output file path (default: `data/logs/random_agent_benchmark.json`)
- `--quiet`: Suppress progress output

The benchmark collects metrics including:
- Success rate
- Mean reward and standard deviation
- Average episode length
- Per-episode detailed metrics

Results are saved to a JSON file for comparison with trained agents.

### Training

Training scripts are under active development.

The first milestone focuses on training a **goal‑conditioned parking agent** using reinforcement learning. Detailed commands and configurations will be documented as each phase is completed.

---

## 📁 Project Structure

```
AiDriver/
├── README.md
├── requirements.txt
├── sanity_run.py
├── run_benchmark.py
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── random_agent.py
│   ├── environments/
│   ├── training/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   └── run_benchmark.py
│   ├── utils/
│   └── config/
│       ├── __init__.py
│       └── env_config.py
│
├── models/
│   ├── parking/
│   ├── parking_obstacles/
│   └── navigation/
│
├── data/
│   ├── maps/
│   └── logs/
│
├── notebooks/
│   └── experiments/
│
├── tests/
└── docs/
```

---

## 🤝 Contributing

Contributions are welcome. This project is intentionally open to experimentation and discussion.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

Please follow PEP‑8 style guidelines and include documentation where appropriate.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📚 Documentation

* [highway‑env Documentation](https://highway-env.readthedocs.io/)
* LinkedIn Series (coming soon)

---

## 🙏 Acknowledgments

* `highway-env` – lightweight driving environments for RL research
* OpenAI Gym / Gymnasium – reinforcement learning interfaces

---

**Note**: This project is developed as a public learning journey. Progress, failures, and design decisions are intentionally shared as part of the process.
