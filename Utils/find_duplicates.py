import os
from collections import defaultdict


def find_duplicate_files(root_dir: str = "project/Data/keywords"):
    file_dict = defaultdict(list)

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(foldername, filename)
            file_dict[filename].append(full_path)

    duplicates = {
        filename: paths for filename, paths in file_dict.items() if len(paths) > 1
    }

    return duplicates


def main():
    duplicates = find_duplicate_files()

    if not duplicates:
        print("No duplicate files found.")
    else:
        with open("duplicates.txt", "w") as file:
            file.write("Duplicate files:\n")
            for filename, paths in duplicates.items():
                file.write(f"{filename}\n")
                for path in paths:
                    file.write(f"{path}\n")

            print("Duplicate files saved to duplicates.txt.")


if __name__ == "__main__":
    main()
