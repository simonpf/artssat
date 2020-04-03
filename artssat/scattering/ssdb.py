import os
import glob

if "SSDB_PATH" in os.environ:
    ssdb_path = os.environ["SSDB_PATH"]
else:
    ssdb_path = "/home/simonpf/src/jupyterhub/data/scattering"

class Shape:
    def __init__(self, name, type = "ice"):
        if type == "ice":
            self.type = "Ice"
        else:
            self.type = "Liquid"

        self.name = name
        #self.path = os.path.join(ssdb_path, "StandardHabits", "FullSet", name + ".xml")
        #self.meta = os.path.join(ssdb_path, "StandardHabits", "FullSet", name + ".meta.xml")
        self.path = os.path.join(ssdb_path, name, name + ".xml")
        self.meta = os.path.join(ssdb_path, name, name + ".meta.xml")


        self.img = os.path.join(ssdb_path, name, "shape_img.png")
        #self.img  = glob.glob(os.path.join(ssdb_path, "SSD", "TotallyRandom", "*", "*", "*",
        #                                   "*" + name.replace("-", "") + "*",
        #                                   "shape_img.png"))[0]

    def copy_to(self, path):
        from shutil import copy
        habit_path = os.path.join(path, self.name)
        if not os.path.exists(habit_path):
            os.mkdir(habit_path)

        copy(self.path, habit_path)
        copy(self.path + ".bin", habit_path)
        copy(self.meta, habit_path)
        copy(self.img, habit_path)

    def __repr__(self):
        return self.name

#standard_habits = glob.glob(os.path.join(ssdb_path, "StandardHabits", "FullSet", "*"))

#names = ["LargePlateAggregate", "6-BulletRosette", "EvansSnowAggregate", "LiquidSphere"]
#for n in names:
#    s = Shape(n)
#    s.copy_to("/home/simonpf/src/jupyterhub/data/scattering")
names = glob.glob(os.path.join(ssdb_path, "*"))
shapes = []
for n in names:
    s = Shape(os.path.basename(n).split(".")[0])
    shapes += [s]


