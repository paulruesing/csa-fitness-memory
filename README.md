# Cuckoo Search with Fitness Memory Implementation

This repository demonstrates a general **cuckoo search algorithm** (CSA) implementation allowing for any fitness
function with an optional **fitness memory instance**. The fitness memory is restricted in size, replacing old entries if 
necessary based on the FIFO-principle. This offers an immense **speed-up particularly with computationally-intensive
fitness functions**.

The code was developed to serve as an efficient way of parametrizing an equivalent circuit model for simulation.

## 1. What is CSA?

CSA is a genetic (population-based) meta-heuristic inspired by the parasitic breeding behavior of cuckoo birds and Lévy
flight behavior of many animals and insects.

The procedure of the optimisation can be described as follows:
- initialise a **random population of k nests** with each one or more eggs (solutions)
- iteratively:
  - **new solutions are calculated using Lévy flight** (serving as a random walk) but only kept if they are improving
  - a **random fraction of nests is abandoned** and replaced if an improving random solution is found
- until final number of iterations is reached

## 2. How does the fitness memory work?
If a fitness memory instance is defined, it is used to speed up the optimization procedure. Precisely, each step
where a fitness values is *read-out*, this is first searched in the fitness memory frame based on the current parameters
and each time fitness is *calculated*, the result is saved in the fitness memory based on the FIFO-principle.

## 3. Repository Structure
- *src/*: source code directory containing classes and methods
- *notebooks/*: jupyter notebooks demonstrating the workflow on a simple illustrative problem

## 4. How to Use
### 4.1. Required Modules
For sole source code usage **pandas**, **numpy**, **matplotlib**, **seaborn** and **tqdm** are required.
For the notebooks **ipython**, **ipywidgets** and **ipykernel** are required.
The recommended way is to install both is calling 
`conda env create -f environment.yml`
in terminal in the project directory.

### 4.2. Recommendations
Usage is demonstrated in the notebook, and it is advised to follow such procedure when implementing.
The utilized class is CuckooSearch from *src/pipeline/cuckoosearch.py*. It can be operated through the following steps:

1. define a **fitness function(s)** to be optimized for. The function receives parameters as a pandas-Series during execution
2. define **boundaries** (lower and upper) for each parameter
3. initialise the **CuckooSearch instance** with boundaries, fitness function, optimization direction and output directory
4. **run the optimisation** for a given number of iterations

## 5. Other Important Information
### 5.1. Authors and Acknowledgment
Paul Rüsing - pr@paulruesing.de - single and main contributor

The CSA formulas and concept are based on
- https://ieeexplore.ieee.org/document/5393690

### 5.2. License
The project is licensed under the MIT license. To view a copy of this license, see [LICENSE](https://github.com/paulruesing/lrp-xai-pytorch?tab=MIT-1-ov-file).
