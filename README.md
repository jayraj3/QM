# QM test
## Instructions
1. Please use the following format to use the code

    ```shell
    python analyze_annotator.py -a "Path to annotator data" -o "Provide operation name" -r "Path to reference data"
    ```
Supported operations are [count_annotator, annotation_time, annotator_work, find_conflict, extra_output, validate_reference_data, annotator_accuracy].
## Installation

This implementation requires the following dependencies (tested on windows 10): 

* Python 3.9 
* [NumPy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/). You can quickly install/update these dependencies by running the following:
  ```shell
  pip install numpy pandas matplotlib
  ```
