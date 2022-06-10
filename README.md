# The Parallelization of complex computational problems

The final product of my 2022 capstone project

To run code found in this project:

- Download and install
  
  -  [Rust](https://www.rust-lang.org/) 
  
  - [Arrayfire](https://arrayfire.com/),
  
  - Your System's latest GPU drivers

- Clone this repo and navigate to ./physsrum-simulation/src/

- Set the `WIDTH`, `HEIGHT`, and the `Window::new` values.
  
  - `WIDTH` and `HEIGHT` represent the area for the agents while the `Window::new` values set the window size.
  
  - *Laptop users* Reduce `NUM_AGENTS` to less than 50_000, unless you know your computer can handle more.

- Compile and run the program by executing  `cargo run --release` in your preferred shell.



To read the paper navigate to ./Paper/
