import torch

from archs.iart_v2_arch import IARTv2 as IART

if __name__ == "__main__":
    # Set up device
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model
    model = IART(in_channels=3, mid_channels=16, embed_dim=32, num_frames=3).to(device)

    # Print model summary (optional, to check the architecture)
    # summary(model, input_size=(3, 3, 64, 64))  # Input shape is (num_frames, channels, height, width)
    data = torch.rand(1, 3, 3, 64, 64).to(device)

    # Run the profiler
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        use_cuda=True
    ) as prof:
        for step in range(5):
            with torch.profiler.record_function("model_inference"):
                model(data)
            # Step the profiler
            prof.step()

    # Print out a summary of the results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

