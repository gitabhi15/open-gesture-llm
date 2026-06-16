# Open-Gesture-LLM: Context-Aware Wearable Gesture Interface

An open-ended wearable interface that maps continuous physical motion to semantic intent. Using an ESP32 and a PyTorch 1D CNN for gesture tokenization, it streams motion primitives to a Large Language Model (LLM) reasoning engine.

This shifts gesture control from fixed, hardcoded actions to dynamic, context-aware human-AI communication.

---

## The Core Concept

Most wearable gesture controllers rely on strict, one-to-one mappings (e.g., a right swipe always means **"Next Slide"**). This project explores a different paradigm:

### Motion as Language

Instead of mapping a gesture directly to an OS command, the hardware simply tokenizes the physical movement and passes it to an LLM.

The LLM acts as the reasoning engine, analyzing the user's current context to determine intent.

```text
Swipe Right + Presentation Mode → NEXT_SLIDE

Swipe Right + Media Mode → NEXT_TRACK
```

---

## Current Hardware & Software Stack

### Hardware

* ESP32 Microcontroller
* ADXL345 Accelerometer
* Wireless UDP streaming over Wi-Fi

### Machine Learning (Perception)

* PyTorch
* Custom 1D Convolutional Neural Network (CNN)

### LLM (Reasoning)

* Google Gemini 2.5 Flash API (`google-genai`)

### Application Layer

* Python
* Linux desktop automation via `xdotool`
* Python subprocess integration

---

## How the Current Architecture Works

The project is currently built on a decoupled microservice architecture running locally on Linux (Fedora).

### 1. Sensor Stream

An ESP32 strapped to the wrist streams raw X, Y, Z acceleration data over Wi-Fi, eliminating physical cable artifacts.

### 2. The Brain (`gesture_llm_engine.py`)

* Uses a **Push-to-Gesture** manual trigger (pressing `ENTER` in the terminal) to record a 2-second motion window.
* Feeds the resulting 50-timestep array into the PyTorch CNN.
* Classifies the motion into one of four gesture tokens:

  * Swipe Right
  * Swipe Left
  * Swipe Up
  * Rotate Right
* Packages the gesture token and active system context into a prompt.
* Calls the Gemini API to infer the user's intended action.

### 3. The Hands (`desktop_controller.py`)

* Listens on a local `localhost` port for the LLM's final decision.
* Uses `xdotool` to simulate physical keyboard events.
* Controls the active desktop window through actions such as:

  * Spacebar
  * Arrow Keys
  * Other context-dependent shortcuts

---

## Setup and Installation

### Prerequisites

* Python 3.11+
* Linux OS with `xdotool` installed
* Google AI Studio API Key (Free Tier)

Install `xdotool`:

**Fedora**

```bash
sudo dnf install xdotool
```

**Debian / Ubuntu**

```bash
sudo apt install xdotool
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MotionLLM.git
cd MotionLLM

python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch numpy scipy google-genai
```

### 3. Configure Gemini

Add your Gemini API key to:

```python
gesture_llm_engine.py
```

### 4. Flash the ESP32

Flash the provided UDP streaming firmware to the ESP32.

### 5. Start the Application Listener

Terminal 1:

```bash
python desktop_controller.py
```

### 6. Start the Inference Engine

Terminal 2:

```bash
python gesture_llm_engine.py
```

---

## Current Limitations & Roadmap

The project is currently in active development.

The Phase 3 prototype successfully validates the core LLM reasoning hypothesis, but several bottlenecks remain.

### 1. Manual "Push-to-Gesture" Trigger

**Current State**

To avoid the challenges of continuous temporal segmentation and noisy sensor data, the system currently requires the user to press `ENTER` before recording a gesture window.

**Next Step**

* Implement a DSP preprocessing pipeline.
* Add a **Null/Idle** gesture class.
* Enable continuous, always-on gesture detection.

---

### 2. API Rate Limits & Latency

**Current State**

The cloud-based Gemini API introduces:

* Network latency
* Quota limitations
* Occasional HTTP 429 errors during rapid interaction

**Next Step**

Replace Gemini API calls with a quantized, locally hosted Small Language Model (SLM) using:

* Ollama
* Local inference pipeline

---

### 3. Sensor Limitations

**Current State**

The ADXL345:

* Is susceptible to motion noise
* Measures only linear acceleration
* Lacks orientation awareness

**Next Step**

Upgrade to a 9-DoF IMU such as:

* BNO085

Benefits:

* Hardware-level sensor fusion
* Absolute 3D orientation tracking
* Improved gesture fidelity

---

## License

Released under the MIT License.
