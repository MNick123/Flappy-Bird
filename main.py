import torch

def main():
    print("Hello from dqncuda!")
    import torch
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Built with CUDA:", torch.backends.cuda.is_built())




if __name__ == "__main__":
    main()
