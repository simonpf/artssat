from IPython import get_ipython

ip = get_ipython()
if not ip is None:
    ip.magic("%matplotlib inline")

import matplotlib.pyplot as plt
import numpy as np

#
# Load stylesheet
#

import os
import os.path
import parts
parts_path = os.path.dirname(parts.__file__)
plt.style.use(os.path.join(parts_path, "..", "misc", "notebook_style_sheet"))
