# Simple Python UGV Simulator
A simple simulator for a 2D ground robot.

Powered by [pygame](https://www.pygame.org/news), this simulator is meant to be a lightweight package to quickly start build your own project.

# Installation Instructions
Clone the repo and install the required packages:
```console
git clone https://github.com/jacobhiggins/python_ugv_sim.git
cd python_ugv_sim
pip install -r .\requirements.txt
```
# Running the Simulator
To run the basic example, from the repo root directory:
```console
python main.py
```

This launches a pygame display with a [differential drive robot](https://en.wikipedia.org/wiki/Differential_wheeled_robot) in the lower-left corner. Use the keyboard to drive the robot:
 - UP/DOWN: Positive/negative forward velocity
 - LEFT/RIGHT: Pivot left/right

 ![Demo of the simulator](./media/demo.gif)