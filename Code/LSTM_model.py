# %%
import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import pandas as pd
#from google.colab import drive
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import model_selection
import matplotlib.pyplot as plt
from IPython.display import display

# %%
runs = [
    {"name": "COVID_only", "df_econ": "df_econ_w_state_noecon.csv", "df_covid": "df_covid_w_state_7_14_21.csv", 
        "params": {'hidden_layer_size': 160, 'num_layers': 2, 'records_length': 147}},
    {"name": "COVID_w_7_14_21", "df_econ": "df_econ_w_state.csv", "df_covid": "df_covid_econ_7_14_21.csv",
        "params": {'hidden_layer_size': 160, 'num_layers': 2, 'records_length': 147}}
]

# 6*3 + 4*3 + 2*3 = 36 trials
#all_parameters = [{"hidden_layer_size": [20, 40], "num_layers": [1, 2, 3], "records_length": [30, 90, 154]},
#              {"hidden_layer_size": [80, 160], "num_layers": [1, 2], "records_length": [30, 90, 154]},
#              {"hidden_layer_size": [320, 500], "num_layers": [1], "records_length": [30, 90, 154]}]

#param_grid = model_selection.ParameterGrid(all_parameters)
#print(param_grid)

all_results = pd.DataFrame(columns=["hidden_layer_size", "num_layers", "records_length", "r2_spend_all", "r2_revenue_all", "r2_emp_combined"])

for run in runs:

    df_econ = pd.read_csv(run["df_econ"], index_col=0)
    df_covid = pd.read_csv(run["df_covid"], index_col=0)


# %%
### PREPARE DATA FOR TORCH
# 1. Set proper indices and sort by state then date
# 2. Scale Data
    train_date = "2020-07-31"

    print(df_econ.shape)
    df_econ_state = df_econ
    df_econ_state.reset_index(inplace=True)
    df_econ_state["Date"] = pd.to_datetime(df_econ_state["Date"], format="%Y-%m-%d")
    df_econ_state.set_index(["statefips","Date"], inplace=True)
    df_econ_state.sort_index(inplace=True)
    print(df_econ_state.columns)

    print(df_econ_state.index.levels[0].dtype)
    print(df_econ_state.shape)

    df_covid_state = df_covid
    df_covid_state.reset_index(inplace=True)
    df_covid_state["Date"] = pd.to_datetime(df_covid_state["Date"], format="%Y-%m-%d")
    df_covid_state.set_index(["statefips","Date"], inplace=True)
    df_covid_state.sort_index(inplace=True)

    df_econ_condensed = df_econ_state.filter(["spend_all", "revenue_all", "emp_combined"], axis=1)

    df_inputs = df_covid_state.copy() #.loc[1]
    df_targets = df_econ_condensed.copy() #.loc[1]
    display(df_inputs)

    df_inputs = df_inputs.loc[df_targets.index]

    inputs = df_inputs.to_numpy()

    stdscaler = StandardScaler()

    idx = pd.IndexSlice
    stdscaler.fit(df_inputs.loc[idx[:, :train_date], :].to_numpy())
    inputs = stdscaler.transform(inputs)

    #inputsAL_normalized = scaler.fit_transform(inputsAL_concat)
    display(inputs)

    targets = df_targets.to_numpy()

    stdscaler.fit(df_targets.loc[idx[:, :train_date], :].to_numpy())
    targets = stdscaler.transform(targets)
    display(df_targets)


# %%
### SET UP THE LSTM TORCH MODEL
    # See https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
    # and https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

    class Model(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1, device=None):
            super().__init__()

            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_size=hidden_layer_size, num_layers=num_layers)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size).to(device),
                                torch.zeros(num_layers,1,self.hidden_layer_size).to(device))
            
            self.n_layers = num_layers

            self.device = device

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), (self.hidden_cell[0].detach(), self.hidden_cell[1].detach()))
            #lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)

            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]
        
        def init_hidden(self, batch_size):
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            # We'll send the tensor holding the hidden state to the device we specified earlier as well
            self.hidden_cell = (torch.zeros(self.n_layers,1,self.hidden_layer_size).to(self.device),
                                torch.zeros(self.n_layers,1,self.hidden_layer_size).to(self.device))
            


    # %%
### INSTANTIATE THE MODEL AND ENCODE THE TARGET VARIABLE
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"

    device = torch.device(dev)

    for training_col in ["spend_all", "revenue_all", "emp_combined"]:
        means = df_targets.groupby('statefips')[training_col].mean()
        temp = df_inputs.index.get_level_values(0).map(means)
        inputs_encoded = np.concatenate((inputs, temp.to_numpy().reshape(-1,1)), axis=1)

        torchInputs = torch.from_numpy(inputs_encoded).float().to(device)
        torchTargets = torch.from_numpy(targets[:,df_targets.columns.get_loc(training_col)]).view((-1,1)).float().to(device)

        #best_r2 = float('-inf')
        #best_params = None

        #for params in param_grid:
        params = run["params"]
        print()
        print("#############################################################################")
        print("Beginning RUN {0} with parameters:".format(run["name"]))
        print(params)

        # Instantiate the model with hyperparameters
        model = Model(input_size=torchInputs.shape[1], output_size=torchTargets.shape[1], hidden_layer_size=params["hidden_layer_size"], num_layers=params["num_layers"], device=device)

        # We'll also set the model to the device that we defined earlier (default is CPU)
        model.to(device)

        # Define hyperparameters
        n_epochs = 32
        lr = 0.1

        # Define Loss, Optimizer
        loss_function = nn.MSELoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.3, weight_decay=0.0001, dampening=0.0005)
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=1e-4, cooldown=2)



        # %%
### ADVANCE THROUGH EPOCHS
# 1. Determine how many records to feed model and therefore number of batches
# 2. Shuffle the order of states every epoch
# 3. Feed in batches and backpropogate from loss function
        records_to_provide = params["records_length"]

        only_texas = False

        train_size = 50
        idx = pd.IndexSlice

        states = df_inputs.index.unique(level=0)
        #states = states[0:5]
        #dates = df_inputs.index.unique()
        dates = df_inputs.index.unique(level=1)

        epoch_losses = []


        

        for epoch in range(n_epochs):
            #optimizer.zero_grad()
            model.init_hidden(1)
            full_loss = 0
            input_locations = 0
            #row_index = 0

            state_i = 0
            dates_counted = 0
            states_shuffled = states.to_numpy().copy()
            np.random.shuffle(states_shuffled)

            #np.random.shuffle(states)

            for state in states_shuffled:
                
                row_index = df_inputs.index.get_loc(state).start
                #assert row_index == row_index2

                records_to_provide=min(params["records_length"], df_inputs.index.get_loc((state, train_date))-1-row_index)

                #for date_index in range(records_to_provide, min(torchInputs.shape[0], train_size)):
                date_count = df_inputs.loc[state].shape[0]
                #date_j = 0
                dates_iterator = np.arange(date_count)

                if ((not only_texas) or state == 48):
                #print(row_index)
                #print(df_inputs.iloc[row_index])
                #np.random.shuffle(dates_iterator)
                    model.init_hidden(1)

                    for date_j in range(records_to_provide, date_count):
                        if (date_j < records_to_provide):
                            continue
                        #elif (date_j > train_size+records_to_provide):
                        #  break
                        elif (df_inputs.loc[state].index[date_j] >= np.datetime64(train_date)):
                        #print("Breaking")
                            break

                        input_locations = np.arange(row_index + date_j - records_to_provide, row_index + date_j)

                        optimizer.zero_grad(set_to_none=True)
                        model.init_hidden(1)

                        y_pred = model(torchInputs.narrow(0,row_index + date_j - records_to_provide, records_to_provide).view(records_to_provide,-1))
                        #print("Predicted value for [{0}-{1})".format(row_index + date_j - records_to_provide, row_index + date_j - records_to_provide+records_to_provide))

                        # Compare to target of day 0
                        single_loss = loss_function(y_pred, torchTargets[row_index + date_j-1,:])
                        #print("Target for {0}".format(row_index + date_j-1))
                        #print(torchTargets[row_index + date_j-1,:].float())
                        
                        single_loss.backward()
                        optimizer.step()

                        input_locations += 1
                        dates_counted += 1

                        full_loss += single_loss.item()

                state_i += 1

            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step(full_loss / (dates_counted))

            if (epoch==0):
                print("Stopped after {0} dates".format(dates_counted))

            if epoch%5 == 0:
                print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}, lr:{curr_lr:1.2e}')
                print("Average Loss over Time Series: {0}".format(full_loss / (dates_counted)))

            epoch_losses.append(full_loss / (dates_counted))
            

        print(f'epoch: {epoch:3} loss: {full_loss:10.10f}, lr:{curr_lr:1.2e}')


        # %%
        #plt.plot(np.arange(len(epoch_losses)), epoch_losses)


        # %%
### HELPER FUNCTION TO DO PREDICTIONS FROM MODEL
        def predict_ts(model, training_col, backwards=False, test_count=30, only_texas=False):
            test_outputs = {training_col: []}
            test_pred_outputs = {training_col: []}
            test_count = 30

            predicted_outputs = torch.zeros([states.size, test_count, targets.shape[1]], device=model.device)

            state_i = 0

            model.eval()
            for state in states:
                if (backwards):
                    test_index = df_inputs.index.get_loc((state, np.datetime64(train_date)+np.timedelta64(-test_count, 'D')))
                else:
                    test_index = df_inputs.index.get_loc((state, train_date))

                row_index = df_inputs.index.get_loc(state).start
                records_to_provide=min(params["records_length"], df_inputs.index.get_loc((state, train_date))-1-row_index)


                #feed_inputs = torch.zeros([records_to_provide,targets.shape[1]], device=model.device, requires_grad=False)
                if ((not only_texas) or state == 48):
                    model.init_hidden(1)

                    for i in range(row_index+records_to_provide-1,test_index):
                        with torch.no_grad():
                            predictions = model(torchInputs[np.arange(i-records_to_provide+1, i+1)])
                            #feed_inputs = torch.roll(feed_inputs, -1, dims=0)
                            #feed_inputs[-1,:]=predictions

                    for i in range(test_count):
                        with torch.no_grad():    
                            model.init_hidden(1)
                            test_inputs = torchInputs[np.arange(test_index+i-records_to_provide+1, test_index+i+1),:]

                            predictions = model(test_inputs.reshape([records_to_provide,-1]))
                            predicted_outputs[state_i,i,:] = predictions

                            for col in test_outputs:
                                test_outputs[col].append(torchTargets[test_index+i, 0].item())

                c=0
                for col in test_outputs:
                    test_pred_outputs[col].extend(predicted_outputs[state_i,:,c].reshape((-1,)).cpu().tolist())
                    c+=1

                #test_index += date_count
                state_i += 1

            return test_outputs, test_pred_outputs

        # %%
        from sklearn.metrics import r2_score, explained_variance_score

        # %%
### GENERATE TEST RESULTS
        test_outputs, test_pred_outputs = predict_ts(model=model, backwards=False, test_count=30, only_texas=only_texas, training_col=training_col)


        # %%
### PRINT TEST RESULTS
        new_row = pd.Series(name=run["name"], data={"hidden_layer_size": params["hidden_layer_size"], "num_layers": params["num_layers"], "records_length": params["records_length"],
                                        "r2_spend_all":0.0, "r2_revenue_all": 0.0, "r2_emp_combined": 0.0})

        for col in test_outputs:
            r2 = r2_score(test_outputs[col], test_pred_outputs[col])
            print("TEST R2 ({0}) = {1:2.6f}".format(col, r2))
            new_row["r2_{0}".format(col)] = r2

        #all_results = all_results.append(new_row, ignore_index=False)

#all_results.to_csv("all_results.csv")
