# MIMOSAS #

MIMOSAS (Multimodal Input Model Output Security Analysis Suite) is a supervised machine learning pipeline developed for classification of multimodal data to inform nuclear security and proliferation detection scenarios. MIMOSAS provides an end-to-end data processing workflow, from data ingestion and pre-processing to model training and test set classification. The pipeline is specified via an input deck, making workflow customization effortless, and the framework is modular allowing for the easy addition of new learning algorithms.

In the current build, the user selects from decision tree, random forest, and feed-forward neural network classifiers to train customizable models with built-in cross validation methods for hyperparameter optimization. Trained model outputs are stored with the associated metadata for rapid deployment. These can be applied in supervised classification to assess previously unseen data or for further training as new observations are added to the existing data set. MIMOSAS provides the capability to fuse a wide range of data sources (e.g., radiation, environmental, acoustic, seismic, imagery, etc.) to make, confirm, and correlate machine learning predictions for nuclear security applications.

Detailed documentation is available in the **MIMOSAS Manual**. Visit https://complexity.berkeley.edu/mimosas/ for more information.

### Prerequisites ###
Python 3 is required to run MIMOSAS, along with the following Python packages: 
- numpy,
- scipy,
- pandas,
- scikit-learn, 
- pytorch, and
- keras.

### Execution ###

To run the software using the default config file: 
```
python main.py
```
To run the software using a custom config file: 
```
python main.py --config custom.config
```

### Documentation ###

MIMOSAS uses Doxygen to generate documentation. To compile the documentation, install Doxygen and issue the following command from the main directory:
```
doxygen Doxyfile
```
The folder *docs* contains all the documentation source pages. These are also hosted via pages at https://nonproliferation.github.io/mimosas/. 

### License ###

Copyright (C)2019-. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

Created by Jared Zhao, Bethany L. Goldblum, Christopher Stewart, Alicia Ying-Ti Tsai, Shruthi Chockkalingam, and Pedro Vicente Valdez, Department of Nuclear Engineering, University of California, Berkeley.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

### Acknowledgements and Disclaimer ###

The project was funded by the U.S. Department of Energy, National Nuclear Security Administration, Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D). 

This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or limited, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.