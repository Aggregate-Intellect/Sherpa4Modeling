from output_transformer import ClassNameOutputTransform
import argparse
import os


def main(args):
    parser = ClassNameOutputTransform()
    filenames = os.listdir(args.input_folder)

    # create output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in filenames:
        with open(os.path.join(args.input_folder, filename), "r") as f:
            output = f.read()
        domain_model = parser.transform_output(output)

        with open(os.path.join(args.output_folder, filename), "w") as f:
            f.write(str(domain_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform output")
    parser.add_argument("--input_folder", type=str, help="Input folder")
    parser.add_argument("--output_folder", type=str, help="Output folder")

    args = parser.parse_args()
    main(args)
