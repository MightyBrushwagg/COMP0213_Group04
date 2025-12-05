"""
Main entry point for the robotic grasping simulation and machine learning pipeline.

This script provides a command-line interface for:
1. Running PyBullet simulations to collect grasping data
2. Training machine learning models to predict grasp success

Usage examples:
    # Run simulations
    python main.py --mode run --object cube --gripper two_finger --iterations 1000
    
    # Train a model
    python main.py --mode train --model logistic_regression --file_save cube-two_finger-data.csv
"""

# Available object types for simulation
objs = ["cube", "cylinder"]

# Available gripper types for simulation
gripper_dic = ["two_finger", "new_gripper"]

# Available machine learning models
models = ["logistic_regression", "svm", "forest", "all"]

if __name__ == "__main__":

    from Simulation.simulation import Simulation
    from Data.data import Data
    from Models.models import Logistic_Regression, SVM, Random_Forest, compare_models
    import argparse
    
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description="Robotic grasping simulation and ML training pipeline"
    )
    
    # Simulation parameters
    parser.add_argument("--object", type=str, default="cube", choices=objs,
                       help="Type of object to use in simulation.")
    parser.add_argument("--gripper", type=str, default="two_finger", choices=gripper_dic,
                       help="Type of gripper to use in simulation.")
    parser.add_argument("--visuals", type=str, default="no visuals",
                       choices=["visuals", "no visuals"],
                       help="Whether to show simulation visuals (slower but visual).")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="Number of simulation iterations to run.")
    parser.add_argument("--save_data", type=bool, default=True,
                       help="Whether to save the simulation data to a CSV file.")
    parser.add_argument("--file_save", type=str, default=None,
                       help="File name to save simulation data (auto-generated if None).")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="run", choices=["run", "train"],
                       help="Operation mode: 'run' for simulations, 'train' for ML training.")
    
    # ML training parameters
    parser.add_argument("--model", type=str, default="logistic_regression", choices=models,
                       help="Type of model to train (or 'all' to compare all models).")
    parser.add_argument("--train_points", type=int, default=120,
                       help="Number of training data points.")
    parser.add_argument("--test_points", type=int, default=300,
                       help="Number of testing data points.")
    parser.add_argument("--val_points", type=int, default=0,
                       help="Number of validation data points.")

    args = parser.parse_args()

    # Execute based on selected mode
    if args.mode == "run":
        # Run simulation mode: collect grasping data
        print("Running simulations...")
        sim = Simulation(args.iterations, object=args.object, gripper=args.gripper,
                       visuals=args.visuals, file_save=args.file_save)
        sim.run_simulations(save=args.save_data)
        
    elif args.mode == "train":
        # Training mode: train ML models on collected data
        print("Training model...")
        data = Data()
        # Load data from file (use provided filename or auto-generate)
        data_file = args.file_save if args.file_save is not None else f"{args.object}-{args.gripper}-data.csv"
        data.import_data(data_file)
        
        # Create and train selected model(s)
        if args.model == "logistic_regression":
            model = Logistic_Regression(data, train_points=args.train_points,
                                       test_points=args.test_points)
            model.fit()
            print(f"{args.model} test accuracy: {model.test():.2f}")
            
        elif args.model == "svm":
            model = SVM(data, train_points=args.train_points, test_points=args.test_points)
            model.fit()
            print(f"{args.model} test accuracy: {model.test():.2f}")
            
        elif args.model == "forest":
            model = Random_Forest(data, train_points=args.train_points,
                                test_points=args.test_points)
            model.fit()
            print(f"{args.model} test accuracy: {model.test():.2f}")
            
        elif args.model == "all":
            # Compare all models
            models_list = [
                Logistic_Regression(data, train_points=args.train_points,
                                  test_points=args.test_points),
                SVM(data, train_points=args.train_points, test_points=args.test_points),
                Random_Forest(data, train_points=args.train_points,
                            test_points=args.test_points)
            ]
            results = compare_models(models_list, data)
            # Print comparison results
            for model_name, accuracy in results.items():
                print(f"{model_name} test accuracy: {accuracy:.2f}")

    # sim = Simulation(1000, object="cube", gripper="two_finger", visuals="no visuals")
    """object = cube, cylinder | gripper = new_gripper, two_finger | visuals = visuals, no visuals """
    # sim.run_simulations(save=False)
    # # sim.save_data(save=True)
    
    # data = sim.data

    
    # data = Data()
    # data.import_data("cylinder-new_gripper-data.csv")
    # # data.remove_nans()
    # data.visualise_data()
    # data.statistics()
    # logr = Logistic_Regression(data, train_points=120, test_points=300)
    # logr.fit()
    # print(logr.test())
    # print(logr.predict(logr.test_data[["x", "y", "z", "roll", "pitch", "yaw"]]))
    # print(logr.test_data["success"])





