raise NotImplementedError

# import subprocess

# from pairo_butler.utils.tools import UGENT, pretty_string


# def get_cuda_packages():
#     command = "dpkg -l | grep cuda"
#     result = subprocess.run(command, shell=True, text=True, capture_output=True)
#     output = result.stdout

#     packages = []
#     for line in output.split("\n"):
#         while "  " in line:
#             line = line.replace("  ", " ")
#         try:
#             packages.append(line.split(" ")[1])
#         except IndexError:
#             pass

#     return packages


# def remove_package(name):
#     subprocess.run("clear", shell=True)
#     print(pretty_string(f"{name}:", color="GREEN"))
#     apt_command = f"sudo apt -y remove {name}"
#     remove_command = f"sudo dpkg --remove --force-remove-reinstreq {name}"
#     purge_command = f"sudo dpkg --purge {name}"

#     for cmd in (apt_command, remove_command, purge_command):
#         result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#         print(result.stdout)

#     subprocess.run("clear", shell=True)


# def main():
#     packages = get_cuda_packages()
#     for package in packages:
#         remove_package(package)

#     result = subprocess.run(
#         "sudo apt clean", shell=True, capture_output=True, text=True
#     )
#     print(result.stdout)
#     result = subprocess.run(
#         "sudo apt update", shell=True, capture_output=True, text=True
#     )
#     print(result.stdout)
#     subprocess.run("clear", shell=True)
#     command = "dpkg -l | grep cuda"
#     result = subprocess.run(command, shell=True, text=True, capture_output=True)
#     print(result.stdout)


# if __name__ == "__main__":
#     main()
