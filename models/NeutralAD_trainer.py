import time
import torch
import numpy as np
from utils import format_time
class NeutralAD_trainer:

    def __init__(self, model, loss_function, device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def _train(self,train_loader, optimizer):

        self.model.train()

        loss_all = 0
        for data in train_loader:
            try:
                samples, _ = data
            except:
                samples = data

            z = self.model(samples)

            loss = self.loss_fun(z)
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()

        return loss_all.item()/len(train_loader.dataset)


    def anomaly_scores(self, loader):
        model = self.model
        model.eval()

        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data
                z= model(samples)
                score = self.loss_fun(z,eval=True)
                score_all.append(score)
        try:
            score_all = np.concatenate(score_all)
        except:
            score_all = torch.cat(score_all).cpu().numpy()

        return score_all


    def train_infer(self, train_loader,cls = None,max_epochs=100, optimizer=None, scheduler=None,
                    test_loader=None, logger=None, log_every=10):

        score = None

        time_per_epoch = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_loss = self._train(train_loader, optimizer)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss}'

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)


        score = self.anomaly_scores(test_loader)
        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())
        elapsed = format_time(avg_time_per_epoch)

        return score