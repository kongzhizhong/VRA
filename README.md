## 📁 File Descriptions

* `main.py`: The main execution script for the project.
* `adaptive_sampling.py`: Implements the adaptive sampling methodology used in the experiments.
* `2d_0.pkl`: An initial data file used for training the ground state, obtained via a fitting process.

---

## ⚙️ Usage Examples

### Example 1: Calculate Ground State for a Single $\Omega$ Value

This example calculates the ground state for a single $\Omega = 0.5$ and reproduces the results shown in **Fig. 4**.

1.  Open **`main.py`** and navigate to lines 28-30.
2.  Set the parameters as follows:
    ```python
    o_min = 0.5
    o_max = 0.5
    num1 = 1
    ```
3.  Run the script:
    ```bash
    python main.py
    ```

---

### Example 2: Calculate Ground States for Multiple $\Omega$ Values

This example enables simultaneous calculation for a range of $\Omega$ values (from 0.58 to 0.62) and reproduces the results shown in **Fig. 9**.

1.  Open **`main.py`** and navigate to lines 28-30.
2.  Set the parameters as follows: (If you set num1 = 4, the script will calculate results at four equally spaced points in the range [o_min, o_max])
    ```python
    o_min = 0.58
    o_max = 0.62
    num1 = 2
    ```
3.  Run the script:
    ```bash
    python main.py
    ```
---

### Example 3: Adaptive Sampling Training

This example runs the standalone adaptive sampling training process.

1.  Run the `adaptive_sampling.py` script:
    ```bash
    python adaptive_sampling.py
    ```
