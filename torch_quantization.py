'''
torch.fx for automatic quantization. Unfortunately by Nov 2024, torchscript (torch.jit) and torch.export does not support very well to the model compilation
'''
import torch
import torch_tensorrt
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, 
    get_symmetric_quantization_config,
)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

def calibrate(model, data_loader):
    torch.ao.quantization.allow_exported_model_train_eval(model)
    model.eval()
    # pdb.set_trace()
    with torch.no_grad():
        for input_ in tqdm(data_loader, desc='calibrating'):
            model(input_)

if __name__ == '__main__':
    model = MyModelDefinition() # Pseudo code 
    model.load_state_dict(static_dict_model, strict=True)
    model.eval()

    example_input = (torch.randn(1, 3, 224, 224),)
    m = torch.export.export_for_training(model, example_input)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

    pt2e = prepare_pt2e(m.module(), quantizer)

    dataset = MyOwnDataset()
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    calibrate(pt2e, dl)
    quantized_m = convert_pt2e(pt2e)

    inputs = [
        torch_tensorrt.Input(
            shape=(1, 8, 480, 480),
            dtype=torch.float32,
        )
    ]
    quantized_ep = torch.export.export(quantized_m, example_input)
    rtmodel = torch_tensorrt.fx.compile(quantized_ep, input=inputs, max_batch_size=1)
    torch.jit.save(rtmodel, export_path)