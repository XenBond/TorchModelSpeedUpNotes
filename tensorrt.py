'''
TensorRT support multiple model compression and acceleration features.
Here's a simple sample to compress the pytorch model into torchscript that support tensorrt
Note that each time you change the gpu arch, you need to re-compile it
'''
import torch
import torch_tensorrt # need to import this when compiling and loading models
import gc 

def ckpt2tensorrt(
    model: torch.nn.Module, # torch modeldefinition
    ckpt_path: str,
    out_path: str,
    device: str = 'cuda:0',
    batch_size: int = 1, # tracing can only support fixed input batch
):
    '''
    ckpt_path: str, the path to the lightning module checkpoint
    out_path: str, the path to save the torchscript model
    '''
    static_dict = torch.load(ckpt_path)['state_dict'] # torch lightning module
    static_dict_model = {k[6:]: v for k, v in static_dict.items() if 'model' in k}
    model.load_state_dict(static_dict_model, strict=True)
    model.eval()
    
    print(f'original model num params: {sum(p.numel() for p in model.parameters()) / 1024 / 1024} BM')
   
    inputs = [
        torch_tensorrt.Input(
            shape=(batch_size, 8, 480, 480),
            dtype=torch.float32,
        )
    ]
    
    # load the state dict
    with torch.no_grad():
        model.eval()
        model.to(device)
        example_input = torch.randn(batch_size, 8, 480, 480).to(device)
        traced_model = torch.jit.trace(model, example_input) # trace for torch script
        rtmodel = torch_tensorrt.ts.compile(
            traced_model, 
            inputs=inputs, 
            truncate_long_and_double=True,
            device=torch_tensorrt.Device(type=torch_tensorrt.DeviceType.GPU, gpu_id=0),
        )
        # traced_model = torch.jit.script(model).cpu() # script the model
        # traced_model.save(out_path)
        torch.jit.save(rtmodel, out_path)
        del model, rtmodel
        del example_input
        torch.cuda.empty_cache()
    return