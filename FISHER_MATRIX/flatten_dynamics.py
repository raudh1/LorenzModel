import torch
import os
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils_saving as us
import utils_plot as up
import utils_nets as un
from lorenz_map import LorenzMap
#from lyapunov_exponents import stable_le, lyapunov_exponent_fulltraj_NN
#from mlps import MLP0, MLP1, MLP2, MLP2_4, MLP3_2,  MLP4_2, MLPs0, MLPs1, MLPs2, MLPs2_2, MLPs2_3, MLPs2_4, MLPs3, MLPs3_2, MLPs4, MLPs4_2, MLPsb0, MLPsb1, MLPsb2, MLPsb2_2, MLPsb2_3, MLPsb2_4, MLPsb3, MLPsb3_2, MLPsb4, MLPsb4_2, MLPs1_16, MLPs1_32, MLPs1_128, MLPs1_1024, MLPs1_3, MLPs1_1024_LINEAR
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device

num_pred = 0
losses = []
lossesloc = []
tol_change = 1e-17

def train(seq, filename, max_epochs, device):
    """
    Create a directory ./filename if it doesn't exist, where filename is a string
    In this directory will be saved :
      Some settings of the model in a file settings.txt
      States of the model and the optimizer
      The plot of the target data
    At the end of training :
      Compute lyapunov exponents at the end of each training
      Create a subdirectory jacobians where it puts graphs of the first singular values of jacobians
      The plot of the final prediction by the model, corresponding to the target data
    """
    print(device)
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)


    us.dir_network(filename, None)
    logfile = filename+"/trainlog.txt"
    lyapfile = filename+"/lyapunovs.txt"
    logfile_epoch = filename+"/loss_epoch.txt"

    seq = seq().double().to(device)
    criterion = nn.MSELoss(reduction='sum')     # and NOT mean ==> ensures consistency with Fisher algebra below
    optimizer = optim.LBFGS(seq.parameters(), lr=0.2)

    #begin to train
    sigma=10.0 ; rho=28.0 ; beta=8 / 3
    FP1 = [np.sqrt(beta * (rho-1)) + 0.1, np.sqrt(beta * (rho-1)), rho-1]
    pos = FP1
#    FP0 = [0, 0.001, 0.0]
#    pos = FP0
#    FP_Robin = [0, 1, 1.05]        
#    pos = FP_Robin
    nb_steps = 3000
    delta_t = 0.01

    Lorenz = LorenzMap(delta_t=delta_t)
    global data_LM
    data = Lorenz.full_traj(nb_steps, pos)
#    nb_steps = 3000 ;  data = data[10000:,:]
    data_LM = data
    data = data[:, np.newaxis, :]

    print(data.shape)
    len_train = data.shape[0]//1
    input  = torch.from_numpy(data[:len_train-1, :, :]).double().to(device)
    target = torch.from_numpy(data[1:len_train, :, :]).double().to(device)

    up.plot_traj(target, "Target", filename)

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps)+" steps, delta t = "+str(delta_t) + ", init pos = " + str(pos)

    us.save_settings(filename, str_settings)
    loss_epoch = []
    
    n_params = sum(p.numel() for p in seq.parameters() if p.requires_grad)      # total nbr weights/biases

    def closure():
        # Fisher information matrix -- LM
        
        sumgradexpcost = np.zeros(n_params)
        sumexpcost     = np.zeros(n_params)        
        cost = np.zeros(len_train-1)
        gradcost = np.zeros((n_params, len_train-1))
        for i in range(len_train-1):
            optimizer.zero_grad()
            out = seq(input[i,:,:])
            targetloc = target[i,:,:]
            targetloc = targetloc[np.newaxis,:,:]
            loss = criterion(out, targetloc)
            loss_np = loss.item()
            loss.backward()
        
            gradcost_theta = np.array([])
#            for param in seq.parameters():
            for param in list(seq.parameters()):
#                weight_or_bias = param.detach()           # extract from nn
                weight_or_bias = param
                w_or_b_grad = weight_or_bias.grad
                w_or_b_grad = w_or_b_grad.view(-1).numpy()    # converts into np vector
                gradcost_theta = np.append(gradcost_theta, w_or_b_grad)

            cost[i] = loss.item()
            gradcost[:,i] = gradcost_theta
            exploss = np.exp(-loss_np)
            sumgradexpcost += exploss * gradcost_theta
            sumexpcost     += exploss
        
#        normaliz = sumgradexpcost / np.log(sumexpcost)
        normaliz = sumgradexpcost / sumexpcost
        Jacob = np.zeros((n_params, n_params))
        for irow in range(len_train-1):
            gradlogLikelirow = -gradcost[:,irow] + normaliz
            Jacob += np.outer(gradlogLikelirow, gradlogLikelirow)        # produces rank1 matrix Jacob            
        
#        InfoFisher = np.linalg.det(Jacob)
        global lamb_Fisher            
        lamb_Fisher = np.linalg.eigvals(Jacob)
        # FIN de Fisher information matrix -- LM

        optimizer.zero_grad()
        out = seq(input)

        loss = criterion(out, target)
#        print('loss:', loss.item())
        us.save_trainlog(logfile, 'loss:' + str(loss.item()))
        loss.backward()

        global losses, lossesloc
        losses.append(loss.item())
        lossesloc = loss.item()

        return loss

    for i in range(1, max_epochs + 1):
        loss_before = 1e10 if i == 1 else losses[-1]
        optimizer.step(closure)
        loss_epoch.append(lossesloc)
        us.save_trainlog(logfile_epoch, str(lossesloc))

        global lamb_Fisher_epochs, info_Fisher
        if i == 1:
            lamb_Fisher_epochs = lamb_Fisher
            info_Fisher = [np.product(lamb_Fisher)]
        else:
#            np.c_[lamb_Fisher_epochs, lamb_Fisher]        # adds a column to lamb_Fisher_epochs
            lamb_Fisher_epochs = np.column_stack((lamb_Fisher_epochs, lamb_Fisher))
            info_Fisher.append(np.product(lamb_Fisher))
        
        print('EPOCH: ', i, '    Training loss: ', lossesloc,'   Info Fisher: ',info_Fisher[i-1],' ',lamb_Fisher_epochs.shape)
        us.save_trainlog(logfile, 'EPOCH: '+str(i))
        torch.save(seq, './'+filename+'/'+filename+'_trained.pt')
        torch.save(seq.state_dict(), './'+filename+'/'+filename+'_training_state.pt')
        torch.save(optimizer.state_dict(), './'+filename+'/'+filename+'_optimizer.pt')

#        if abs(loss_before - losses[-1]) < tol_change or i == max_epochs:
        if i == max_epochs:
            if i == max_epochs:
                us.save_trainlog(logfile, 'Max epoch reached')
            else:
                us.save_trainlog(logfile, 'Converged')

            us.dir_jacobians(filename, "jacobians")
            up.plot_singval1(seq, data, Lorenz, delta_t, nb_steps, 0, 5000, i, filename+"/jacobians")
            up.plot_losses(losses, filename, "all_losses")
            up.plot_traj(seq(input), "Final_pred", filename)

            with torch.no_grad():
                data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                pred = seq(input1)
                loss = criterion(pred, target1)
                print('STEP TEST: ', i, 'test loss:', loss.item())

            if torch.cuda.is_available():
                preds = pred[:, 0, :].cpu().detach().numpy()
            else:
                preds = pred[:, 0, :].detach().numpy()

#            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
#            print("Lyapunov of NN model :", lyap_nn)
#            us.save_lyap(lyapfile, "Epoch "+str(i))
#            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")
            lyap_nn = np.random.rand(3)

            return True, lyap_nn, loss_epoch

        if losses[-1] > 1e10 or np.isnan(losses[-1]):
            print('DIVERGED !!')
            return False, 0, 0

#        if i % 20 == 0:
#            torch.save(seq, './'+filename+'/'+filename+'_trained_epoch'+str(i)+'.pt')
#            with torch.no_grad():
#                data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
#                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
#                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
#                pred = seq(input1)
#                loss = criterion(pred, target1)
#                print('STEP TEST: ', i, 'test loss:', loss.item())
#
#            if torch.cuda.is_available():
#                preds = pred[:, 0, :].cpu().detach().numpy()
#            else:
#                preds = pred[:, 0, :].detach().numpy()
#
##            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
##            print("Lyapunov of NN model :", lyap_nn)
##            us.save_lyap(lyapfile, "Epoch "+str(i))
##            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")
#            lyap_nn = np.random.rand(3)
#
#        if i == 1:
#            data_test = Lorenz.full_traj(100000, pos)
#            lyap_true_lorenz = stable_le(Lorenz.jacobian, data_test, delta_t=delta_t)
#            print("Lyapunov of true Lorenz :", lyap_true_lorenz)
#            us.save_lyap(lyapfile, "True lyapunov :"+str(lyap_true_lorenz)+"\n")

#        print()

def train_loop(seq, filename, max_epochs, device, num_train=3, max_failures=10):
    if os.path.exists(filename):
        shutil.rmtree(filename)
    success = False
    lyaps = np.array([])
    for i in range(num_train):
        print('Dealing w/ model ',filename,' and sample #', i)
        global losses
        losses = []
        trained, lyapsloc, lossloc = train(seq, "training", max_epochs, device)
        lossloc = np.array(lossloc)
        if trained:
            shutil.move("training", filename+"/training"+str(i))
            success = True
#            if lyaps == []:
            if lyaps.size == 0:
                lyaps = lyapsloc[:, np.newaxis]
                loss_epoch = lossloc[:, np.newaxis]                
            else:
                lyaps = np.append(lyapsloc[:, np.newaxis], lyaps, axis=1)
                loss_epoch = np.append(lossloc[:, np.newaxis], loss_epoch, axis=1)
        else:
            loss_epoch = np.append(lossloc[:, np.newaxis], loss_epoch, axis=1)
    print(lyaps)
    if not success:
        failures = num_train
        while not success and failures < max_failures:
            trained = train(seq, "training_nn", max_epochs, device)
            if trained:
                shutil.move("training_nn", filename+"/training"+str(i))
                success = True
                failures += 1
    return lyaps, loss_epoch
    

def full_train():
#    relus = [MLP0, MLP1, MLP2, MLP2_4, MLP3_2, MLP4_2]
#    name_relus = ["nn0", "nn1", "nn2", "nn2_4", "nn3_2", "nn4_2"]
#    swish = [MLPs0, MLPs1, MLPs2, MLPs2_2, MLPs2_3, MLPs2_4, MLPs3, MLPs3_2, MLPs4, MLPs4_2]
#    name_swish = ["nns0", "nns1", "nns2", "nns2_2", "nns2_3", "nns2_4", "nns3", "nns3_2", "nns4", "nns4_2"]
#    swish_wb = [MLPsb0, MLPsb1, MLPsb2, MLPsb2_2, MLPsb2_3, MLPsb2_4, MLPsb3, MLPsb3_2, MLPsb4, MLPsb4_2]
#    name_swish_wb = ["nnsb0", "nnsb1", "nnsb2", "nnsb2_2", "nnsb2_3", "nnsb2_4", "nnsb3", "nnsb3_2", "nnsb4", "nnsb4_2"]
#    nets = relus + swish + swish_wb
#    names = name_relus + name_swish + name_swish_wb

    ndims = 3    # LORENZ

#    nets = [MLPs1_16, MLPs1_32, MLPs1_128, MLPs1_1024]
#    names = ["nns1_16", "nns1_32", "nns1_128", "nns1_1024"]
#    nets = [MLPs1_16]
#    names = ["nns1_16"]
    nets = [MLPs2_2]
    names = ["nns2_2"]

    max_epochs = 20
    max_samples = 1
    nconfigs = np.shape(nets)[0]
#    itab = np.linspace(0, nconfigs-1, nconfigs, dtype='int')
    itab = np.zeros(nconfigs, dtype='int')
    lyapmean = np.zeros([nconfigs,ndims])
    quantl = np.zeros([nconfigs,ndims])
    quantu = np.zeros([nconfigs,ndims])
    loss_net = np.zeros([max_epochs, max_samples, nconfigs])
    ii = -1
    for net, name in zip(nets, names):
        ii += 1
        itab[ii] = ii
        print(name)
        lyapsloc, loss_net[:,:,ii] = train_loop(net, name, max_epochs, device, max_samples)
        # ndims x max_samples
        lyapmean[ii,:] = np.mean(lyapsloc, axis = 1)
        quantl[ii,:] = np.quantile(lyapsloc, 0.025, axis = 1)
        quantu[ii,:] = np.quantile(lyapsloc, 0.975, axis = 1)
#        ii += 1
        
        
    imax = 0        # indice du Lyap exp maxi (first)
    plt.figure()
    postquant = plt.fill_between(itab, quantl[:,imax], quantu[:,imax], alpha=0.15, color='cyan', label='Max Lyapexp')
    postmean = plt.plot(itab, lyapmean[:,imax], color = 'blue')
    
    plt.figure()
    color_net = ['cyan', 'red', 'blue', 'green']
    for inet in range(nconfigs):
        for isamp in range(max_samples):
            plt.loglog(loss_net[:, isamp, inet], color = color_net[inet], alpha = 0.5)



if __name__ == '__main__':
    full_train()

    max_epochs = lamb_Fisher_epochs.shape[1]
    plt.figure()
#    for icol in range(max_epochs):
#        plt.plot(np.log(np.abs(lamb_Fisher_epochs[:,icol])))
    icol = 0            ;  IFish0 = np.log(np.abs(lamb_Fisher_epochs[:,icol])) ; plt.plot(IFish0)
    icol = max_epochs-1 ;  IFishN = np.log(np.abs(lamb_Fisher_epochs[:,icol])) ; plt.plot(IFishN)
    plt.title("Fisher spectrum w/ FP_random and J = "+str(losses[-1]))
    np.savetxt("FP_random_MLPs2_2.txt", np.column_stack((IFish0,IFishN)), fmt="%s")
    
    plt.figure()
    plt.plot(np.log(np.abs(info_Fisher)))
    
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(data_LM[:,0],data_LM[:,1],data_LM[:,2])
