import utils
import sys

def run(directory):
    model = utils.save_directory_to_model(directory, "modelv2_testing.pkl")

if __name__ == "__main__":
    run(sys.argv[1])
