from simulation import Simulation
from data import Data
from models import Logistic_Regression

if __name__ == "__main__":
    # sim = Simulation(1000, object="cylinder")
    data = Data()
    data.import_data("cube-twofingergripper")
    data.remove_nans()
    # data.visualise_data()
    data.statistics()
    logr = Logistic_Regression(data, test_points=200)
    logr.fit()
    print(logr.test())
    print(logr.predict(logr.test_data[["x", "y", "z", "roll", "pitch", "yaw"]]))
    print(logr.test_data["success"])





