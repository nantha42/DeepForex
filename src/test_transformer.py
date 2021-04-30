from transformer import *
from essential_functions import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import ast 
from torchviz import make_dot
from transformer_v2 import *
from numpy import load
import torch.nn as nn
import time


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


if __name__ == '__main__':
    model_name = "EvalMark-II-LearningRate"
    data_file = "raw.npy"
    model = torch.load("../models/"+model_name)
    # make_dot(prediction_model).render("attached", format="png")
    procsd_data = load("../preprocessed/"+data_file)
    model_name = "Mark-II-Layer-1"
    train_data =torch.tensor(procsd_data)[:int(len(procsd_data)*0.70)]
    val_data = torch.tensor(procsd_data)[int(len(procsd_data)*0.70):int(len(procsd_data)*0.90)]
    test_data = torch.tensor(procsd_data)[int(len(procsd_data)*0.90):]

    train_data = train_data.contiguous()
    if torch.cuda.is_available():
        train_data = train_data.to(dev)
        val_data = val_data.to(dev)
        test_data = test_data.to(dev)

    batch_size = 32
    ntokens = 240
    train_data = batchify(train_data,batch_size)
    # print(train_data.shape)
    val_data = batchify(val_data,batch_size)
    test_data = batchify(train_data,batch_size)
    dataLoader = CustomDataLoader(val_data)
    # dataLoader.get_batch_from_batches(0)
    print("Batch Count",dataLoader.batchcount())


    tdata,ttar = dataLoader.get_batch_from_batches(0)
    
    # print(tdata,ttar)
    tdata.shape,ttar.shape
    tdata = tdata.transpose(0,1).contiguous().type(torch.LongTensor)
    tar = ttar.transpose(0,1).contiguous().type(torch.LongTensor)
    tdata.shape,tar.shape


    s_inp = tdata[3].view(1,-1)
    s_tar = tar[3]

    s_tar_inp = s_tar[:-1].view(1,-1)
    s_tar_out = s_tar[1:].view(1,-1)

    
    
    # print(s_inp)
    # print(s_tar_inp)
    model.eval()

    print("S_TAR_INP",s_tar_inp)
    clone_inp = s_tar_inp.clone().detach()
    clone_inp[:,1:] = 0
    print("Clone ",clone_inp)
    src_mask,tar_mask = create_masks(s_inp,clone_inp)
    print(src_mask)
    print(tar_mask)

    for i in range(7): 
        # clone_inp[:,6:] = 0
        with torch.no_grad():
            src_mask,tar_mask = create_masks(s_inp,clone_inp)
            out = model(s_inp,clone_inp,src_mask,tar_mask)
            out = out.view(-1,out.size(-1))
            nexx = torch.argmax(out,dim=1).tolist()
            print("Blone",clone_inp.tolist())
            print("Clone",nexx)
            clone_inp[0,i+1] = nexx[i]
            print("Clone",clone_inp)
        # break;
        
    plot_multiple([clone_inp.tolist()[0],nexx,s_tar_out.tolist()[0]],["input","predicted","actual"])
    # out.shape 
    # out = model(s_inp,s_tar_inp,src_mask,tar_mask)
    # out = out.view(-1,out.size(-1))

    # print(out.shape,s_tar_out.shape)
    # print(torch.nn.functional.cross_entropy(out,s_tar_out[0]-1))
    # print(torch.argmax(out,dim=1))
    # print(s_tar_out-1)
    # print(s_inp.tolist()[0],s_tar.tolist())
    # print(s_inp.tolist()[0]+s_tar.tolist())

    # print("Error    ",s_tar_out.tolist()[0])
    # print("Predicted",torch.argmax(out,dim=1).tolist())
    # previous_seq = s_inp.tolist()[0]
    # previous_seq.append(s_tar.tolist()[0])
    # print(previous_seq)
    # plot_multiple([previous_seq + s_tar_out.tolist()[0],previous_seq+ torch.argmax(out,dim=1).tolist()],legend=["actual","predicted"])
    # actual = s_inp.tolist()[0] + s_tar_out.tolist()
    # predicted = s_inp.tolist()[0] + torch.argmax(out,dim=1).tolist() 
    # print(actual)
    # print(predicted)
    # plot_multiple([actual,predicted],legend=["actual","predicted"])