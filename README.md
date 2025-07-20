# Computer graphics exam
This is the esam repo of computer graphics. The project is a simple airplane simulator 3D game that uses Vulkan for rendering and ODE for physics simulation.

# Game Overview

The game features an airplane that can initially fly freely in an infinite world composed of small lakes and rolling hills filled with colorful trees. The user can then activate **Challenge Mode**, where they have **two minutes to collect 10 gems**.
⚠️ Be careful: crashing into the ground or falling into water ends the game with **no restart** allowed.

---

# Main Features

### 🌍 Procedural Infinite World

* Terrain is procedurally generated using **Perlin noise** to create realistic hills and dynamically place lakes based on elevation.

### 🌲 Dynamic Environment

* Trees are randomly distributed and **updated in real-time** as the plane moves.

### 🔊 Immersive Audio

* Ambient sound effects are provided through the integration of the **OpenAL audio library** for a more immersive experience.

### ✈️ Flight Physics & Collision

* Includes a physics model that handles movement and collisions.
* The plane **automatically corrects roll** to maintain better stability in flight.

### 👀 Multiple Camera Views

* **First Person View** → Press `1`
* **Third Person View** → Press `2`, with selectable perspectives:

    * Right Side View → `E`
    * Left Side View → `Q`
    * Rear View → `X`
* Switch between different **projection types** using: `F1` (perspective), `F2` (orthographic from the top), `F3` (isometric)

### 🎮 Game Modes

* **Free Exploration Mode**
  This mode starts when the user presses `P` from the startup menu.
  Fly freely through the infinite world without time limits or objectives.

* **Challenge Mode**
  While in Free Exploration Mode, the user can press `H` to activate Challenge Mode.
  In this mode, you have **2 minutes to collect 10 gems**. Good luck!

*  **Lose Conditions:**

    * Crashing into terrain
    * Falling into water
    * Time runs out before collecting all gems
* **Win Condition:**

    * Collecting all 10 gems within the time limit

### 💡 Realistic Lighting

* Implements **Cook-Torrance** lighting model.
* Terrain is rendered using:

    * **PBR (Physically Based Rendering)**
    * **Normal maps**
    * **Detail textures**
* Other elements (airplane, trees, gems) are rendered with:

    * **Roughness maps** (generated with Perlin noise)
    * **Metallic textures** (for gems)
* **Lighting Types:**

    * **Direct Light** → Directional
    * **Indirect Light** → Ambient
* **Sky** is implemented using a **cubemap**


# Installation
Download or clone the repository, then create a folder called `externals` in the root directory and download the following libraries:
- [ODE library](https://github.com/thomasmarsh/ODE) for physics simulation.

Then build and compile the project in release mode. The project is built using CMake.