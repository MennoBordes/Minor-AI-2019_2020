# Minor-AI-2019_2020 > AI Pilot

This was a school project where we were tasked with creating a Artificial Intelligence.
\
We had chosen to create an AI pilot.
## Table of contents

1. [Requirements And Installs](#1-requirements-and-installs)
2. [Gym environment](#2-gym-environment)
3. [X-plane setup](#3-x-plane-setup)
4. [X-plane connect setup](#4-x-plane-connect-setup)
5. [Run](#5-run)

# 1 Requirements and Installs
Before continuing please ensure that the python version used in this environment matches the required python version
```
Python == 3.7
```
Also ensure that the requirements.txt file has been executed. This ensures that all required packages will 
automatically get installed.
```
pip install -r requirements.txt
```
In case of running the AI for take off, run the following code:
```
pip install tensorflow numpy matplotlib gym PyDirectInput PyGetWindow
```

# 2 Gym environment
To install the gym_xplane environment run the following code: 
```
cd gym-xplane
pip install -e .
cd ..
```
In case of running the AI for taking off, run the following code:
```
cd AI_takeoff/customGym
pip install -e .
cd ../..
```

# 3 X-plane setup
Please install Xplane from the [Xplane-11](https://www.x-plane.com/) website.
\
It is possible to use the Demo version, although it will result in automatic crashes after 15 minutes of play-time.  

### 3.1 Check if X Plane is set up corectly:
* Under Settings > Data output > General Data output > 
  * Network configuration is turned **ON**
  * IP address  =                   192.168.0.1
  * Port        =                   49000
* Under Settings > Data output > Dataref Read/Write >
  * Networked computer is turned    **ON**
  * IP address  =                   192.168.0.1
  * Port        =                   49000

# 4 X-plane connect setup
Visit [Xplane-connect](https://github.com/nasa/XPlaneConnect/releases) and download **Version 1.2.1**
\
Open the downloaded folder XPlaneConnect-1.2.1 > Open folder xpcPlugin > Select folder XplaneConnect
\
Copy folder **XPlaneConnect** to the Game Install folder > **Resources** > **plugins**

### 4.1 Testing X-plane connect installation
In order to test whether the library has been correctly installed

Start XPlane and load a flight.
\
At the top, click on plugins > plugin admin > enable/disable and verify that **X-Plane Connect [Version 1.2.1]** is present. 

# 5 Run
All files which may need to be accessed below are located in the 
[DQN](https://github.com/Skillerde6de/Minor-AI-2019_2020/tree/master/DQN) folder. 

Since there are multiple AI's which can be chosen, it is important to start at the proper location according for what
is specified in the current_training_model.py file.

* TakeOff
    * Within Xplane, start a new flight from **Schiphol** with the following customization:
        * Starting from the ramp/~~runway~~ 
        * At apron S82R
    * After making sure the proper location has been set, start the flight.
    * Once the flight has finished loading, you can run the ./AI_takeoff/DeepQ/main.py file.

* Cruise:
    * Make sure current_training (_within current_training_model.py_) is set to training_Cruise
    * Within Xplane, start a new flight from **Schiphol** with the following customization:
        * Starting from the ~~ramp~~/runway with a 10 nm approach
        * At runway 18R
    * After making sure the proper location has been set, start the flight.
    * Once the flight has finished loading, you can run the DQN_2.py file.

* Landing: 
    * Make sure current_training (_within current_training_model.py_) is set to training_Landing
    * Within Xplane, start a new flight from **Schiphol** with the following customization:
        * Starting from the ~~ramp~~/runway with a 3 nm approach
        * At runway 27
    * After making sure the proper location has been set, start the flight.
    * Once the flight has finished loading, you can run the DQN_2.py file.

## Authors

* **Hayder Ali**  - [Hayder1999](https://github.com/Hayder1999)
* **Felix de Jonge**  - [FelixdeJonge](https://github.com/FelixdeJonge)
* **Menno Bordes**  - [Skillerde6de](https://github.com/Skillerde6de)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Skillerde6de/Minor-AI-2019_2020/blob/master/LICENSE) file for details
