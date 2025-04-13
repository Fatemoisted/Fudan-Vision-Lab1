import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='程序描述信息')
parser.add_argument('--mode', type=str, default="test", help='指定模式')

class DataLoader:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None
        self.test_data = None
        self.test_labels = None
        
    def load_cifar10(self, data_dir):
        train_files = [f'data_batch_{i}' for i in range(1, 6)]
        test_file = 'test_batch'
        X_train, y_train = [], []
        for file in train_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                X_train.append(entry['data'])
                y_train.extend(entry['labels'])

        
        X_train = np.vstack(X_train).astype(np.float32)
        y_train = np.array(y_train)
        with open(os.path.join(data_dir, test_file), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            X_test = entry['data'].astype(np.float32)
            y_test = np.array(entry['labels'])
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) 
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
        num_train = X_train.shape[0]
        num_val = int(num_train * 0.1)  
        
        indices = np.random.permutation(num_train)
        train_idx, val_idx = indices[num_val:], indices[:num_val]
        
        self.val_data = X_train[val_idx]
        self.val_labels = y_train[val_idx]
        self.train_data = X_train[train_idx]
        self.train_labels = y_train[train_idx]
        self.test_data = X_test
        self.test_labels = y_test

        self._preprocess()
        
    def _preprocess(self):
        mean = np.mean(self.train_data, axis=(0, 1, 2), keepdims=True)
        std = np.std(self.train_data, axis=(0, 1, 2), keepdims=True)

        self.train_data = (self.train_data - mean) / std
        self.val_data = (self.val_data - mean) / std
        self.test_data = (self.test_data - mean) / std
        
    def get_batch(self, split):
        if split == 'train':
            data = self.train_data
            labels = self.train_labels
        elif split == 'val':
            data = self.val_data
            labels = self.val_labels
        elif split == 'test':
            data = self.test_data
            labels = self.test_labels
        
        indices = np.random.choice(data.shape[0], self.batch_size, replace=False)
        batch_data = data[indices]
        batch_labels = labels[indices]
        batch_data = batch_data.reshape(self.batch_size, -1)
        
        return batch_data, batch_labels

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_grad(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Loss:
    @staticmethod
    def cross_entropy(y_pred, y_true):
        N = y_pred.shape[0]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(N), y_true] = 1
        y_pred = np.maximum(y_pred, 1e-15)

        loss = -np.sum(y_one_hot * np.log(y_pred)) / N
        return loss
    
    @staticmethod
    def cross_entropy_grad(y_pred, y_true):
        N = y_pred.shape[0]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(N), y_true] = 1
        grad = (y_pred - y_one_hot) / N
        return grad


class ThreeLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size1)
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.b2 = np.zeros(hidden_size2)
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.b3 = np.zeros(output_size)

        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_grad = Activation.relu_grad
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_grad = Activation.sigmoid_grad
        else:
            raise ValueError("不支持的激活函数类型")

        self.cache = {}
    
    def forward(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.activation(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.activation(z2)
        z3 = a2.dot(self.W3) + self.b3
        a3 = Activation.softmax(z3)
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        self.cache['z3'] = z3
        self.cache['a3'] = a3
        
        return a3
    
    def backward(self, y_true, reg=0.0):
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        a2 = self.cache['a2']
        z3 = self.cache['z3']
        a3 = self.cache['a3']
        N = X.shape[0]

        dz3 = Loss.cross_entropy_grad(a3, y_true)
        dW3 = np.dot(a2.T, dz3) + reg * self.W3
        db3 = np.sum(dz3, axis=0)
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.activation_grad(z2)
        dW2 = np.dot(a1.T, dz2) + reg * self.W2
        db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_grad(z1)
        dW1 = np.dot(X.T, dz1) + reg * self.W1
        db1 = np.sum(dz1, axis=0)
        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
            'W3': dW3,
            'b3': db3
        }
        
        return grads
    
    def save_model(self, filepath):
        model_params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
            'input_size': self.input_size,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'output_size': self.output_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        
        self.W1 = model_params['W1']
        self.b1 = model_params['b1']
        self.W2 = model_params['W2']
        self.b2 = model_params['b2']
        self.W3 = model_params['W3']
        self.b3 = model_params['b3']
        self.input_size = model_params['input_size']
        self.hidden_size1 = model_params['hidden_size1']
        self.hidden_size2 = model_params['hidden_size2']
        self.output_size = model_params['output_size']
        
        print(f"模型已从{filepath}加载")

class Trainer:
    def __init__(self, model, data_loader, learning_rate=1e-3, reg=1e-5):
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.reg = reg
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.best_val_acc = 0
        self.best_model_path = 'best_model.pkl'
    
    def train(self, num_epochs=10, lr_decay=0.95, save=True):
        print("开始训练...")
        num_train = self.data_loader.train_data.shape[0]
        iterations_per_epoch = max(num_train // self.data_loader.batch_size, 1)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_loss = 0
            for i in range(iterations_per_epoch):
                X_batch, y_batch = self.data_loader.get_batch('train')
                scores = self.model.forward(X_batch)
                loss = Loss.cross_entropy(scores, y_batch)
                if self.reg > 0:
                    loss += 0.5 * self.reg * (np.sum(self.model.W1 * self.model.W1) + 
                                             np.sum(self.model.W2 * self.model.W2) +
                                             np.sum(self.model.W3 * self.model.W3))
                
                epoch_loss += loss
                grads = self.model.backward(y_batch, self.reg)
                self.model.W1 -= self.learning_rate * grads['W1']
                self.model.b1 -= self.learning_rate * grads['b1']
                self.model.W2 -= self.learning_rate * grads['W2']
                self.model.b2 -= self.learning_rate * grads['b2']
                self.model.W3 -= self.learning_rate * grads['W3']
                self.model.b3 -= self.learning_rate * grads['b3']

            epoch_loss /= iterations_per_epoch
            self.train_loss_history.append(epoch_loss)
            val_loss, val_acc = self.evaluate(split='val')
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if save:
                    self.model.save_model(self.best_model_path)

            self.learning_rate *= lr_decay
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s, "
                  f"LR: {self.learning_rate:.6f}")
        
        print(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f}")
    
    def evaluate(self, split='val'):
        if split == 'val':
            data = self.data_loader.val_data.reshape(self.data_loader.val_data.shape[0], -1)
            labels = self.data_loader.val_labels
        else:  # 'test'
            data = self.data_loader.test_data.reshape(self.data_loader.test_data.shape[0], -1)
            labels = self.data_loader.test_labels
        scores = self.model.forward(data)
 
        loss = Loss.cross_entropy(scores, labels)
        if self.reg > 0:
            loss += 0.5 * self.reg * (np.sum(self.model.W1 * self.model.W1) + 
                                     np.sum(self.model.W2 * self.model.W2) +
                                     np.sum(self.model.W3 * self.model.W3))
        predictions = np.argmax(scores, axis=1)
        accuracy = np.mean(predictions == labels)
        
        return loss, accuracy
    

class HyperparamSearch:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.results = []
    
    def _evaluate_params(self, params):
        lr, hidden_size1, hidden_size2, reg = params
        
        print(f"尝试参数: lr={lr}, hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, reg={reg}")

        input_size = 32 * 32 * 3
        output_size = 10
        model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)
        trainer = Trainer(model, self.data_loader, learning_rate=lr, reg=reg)
        trainer.train(num_epochs=10, lr_decay=0.95)
        _, val_acc = trainer.evaluate(split='val')

        result = {
            'lr': lr,
            'hidden_size1': hidden_size1,
            'hidden_size2': hidden_size2,
            'reg': reg,
            'val_acc': val_acc,
            'train_loss_history': trainer.train_loss_history,
            'val_loss_history': trainer.val_loss_history,
            'val_acc_history': trainer.val_acc_history
        }
        
        return result
    
    # Define this as a class method instead of a local function
    def _evaluate_and_update(self, params_and_data):
        params, counter, lock = params_and_data
        result = self._evaluate_params(params)
        with lock:
            counter.value += 1
        return result
    
    def run_search(self, num_workers=None):
        learning_rates = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        hidden_sizes1 = [256, 512, 1024, 2048]
        hidden_sizes2 = [256, 512, 1024, 2048]
        reg_strengths = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        param_combinations = []
        
        for lr in learning_rates:
            for hidden_size1 in hidden_sizes1:
                for hidden_size2 in hidden_sizes2:
                    for reg in reg_strengths:
                        param_combinations.append((lr, hidden_size1, hidden_size2, reg))
        
        total_combinations = len(param_combinations)
        print(f"开始并行超参数搜索，共{total_combinations}种组合...")
        
        if num_workers is None:
            num_workers = 1
        with mp.Manager() as manager:
            counter = manager.Value('i', 0)
            lock = manager.Lock()
            task_params = [(params, counter, lock) for params in param_combinations]
            with mp.Pool(processes=num_workers) as pool:
                async_result = pool.map_async(self._evaluate_and_update, task_params)
                pbar = tqdm(total=total_combinations)
                while not async_result.ready():
                    pbar.n = counter.value
                    pbar.refresh()
                    time.sleep(60)
                pbar.n = counter.value
                pbar.refresh()
                pbar.close()
                self.results = async_result.get()
        best_result = max(self.results, key=lambda x: x['val_acc'])
        best_val_acc = best_result['val_acc']
        best_params = {
            'lr': best_result['lr'],
            'hidden_size1': best_result['hidden_size1'],
            'hidden_size2': best_result['hidden_size2'],
            'reg': best_result['reg'],
        }

        with open('hyperparam_search_results_2.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"超参数搜索完成! 最佳参数: {best_params}, 验证准确率: {best_val_acc:.4f}")
        print(f"搜索结果已保存到 'hyperparam_search_results.pkl'")
        
        return best_params, self.results

def main(args):
    np.random.seed(42)
    data_loader = DataLoader(batch_size=64)
    data_dir = 'cifar-10' 
    data_loader.load_cifar10(data_dir)
    input_size = 32 * 32 * 3 
    hidden_size = 2048
    output_size = 10
    model = ThreeLayerNN(input_size, hidden_size, hidden_size, output_size, activation='relu')
    trainer = Trainer(model, data_loader, learning_rate=5e-2, reg=1e-3)
    if args.mode == "train":
        trainer.train(num_epochs=40, lr_decay=0.95)
    elif args.mode == "test":
        model.load_model(trainer.best_model_path)
        test_loss, test_acc = trainer.evaluate(split='test')
        print(f"测试集上的准确率: {test_acc:.4f}")
    else:
        hyperparam_search = HyperparamSearch(data_loader)
        best_params, results = hyperparam_search.run_search(num_workers=9)
        
        final_model = ThreeLayerNN(
            input_size, 
            best_params['hidden_size1'],
            best_params['hidden_size2'],
            output_size,
            activation='relu'
        )
        
        final_trainer = Trainer(
            final_model, 
            data_loader, 
            learning_rate=best_params['lr'], 
            reg=best_params['reg']
        )
        
        final_trainer.train(num_epochs=30, lr_decay=0.95)
        final_test_loss, final_test_acc = final_trainer.evaluate(split='test')
        print(f"最终模型在测试集上的准确率: {final_test_acc:.4f}")
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)