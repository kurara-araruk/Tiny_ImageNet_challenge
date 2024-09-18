import numpy as np
import matplotlib.pyplot as plt
import os
import japanize_matplotlib

import torch


########################################################################################################################


# fit関数の定義
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history, scheduler=None):

    # tqdmのインポート
    from tqdm import tqdm

    base_epochs = len(history)

    for epoch in range(base_epochs, base_epochs+num_epochs):
        n_train_acc, n_val_acc = 0,0
        train_loss, val_loss = 0,0
        n_train, n_test = 0,0

        # 訓練フェーズ
        net.train()
        for inputs, labels in tqdm(train_loader):
            train_batch_size = len(labels)
            n_train += train_batch_size

            # GPUへ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 学習
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.max(outputs, 1)[1]

            # 記録用
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()


        # 推論フェーズ
        net.eval()
        with torch.no_grad():
            for inputs_test, labels_test in tqdm(test_loader):
                test_batch_size = len(labels_test)
                n_test += test_batch_size

                # GPUヘ転送
                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)

                # 推論
                outputs_test = net(inputs_test)
                loss_test = criterion(outputs_test, labels_test)
                predicted_test = torch.max(outputs_test, 1)[1]

                # 記録用
                val_loss +=  loss_test.item() * test_batch_size
                n_val_acc +=  (predicted_test == labels_test).sum().item()


        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        # 記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))

        if scheduler is not None:
            scheduler.step()

    return history


########################################################################################################################


# 学習ログ出力関数の定義
def evaluate_history(history, save_name, save_dir):
    # 保存先フォルダが存在しない場合、作成する
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}')
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}')

    num_epochs = len(history)
    unit = num_epochs / 10

    # 損失曲線の表示
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('損失曲線')
    plt.legend()
    plt.savefig(f'{save_dir}/{save_name}_loss.jpg')
    plt.close()

    # acc曲線の表示
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('精度曲線')
    plt.legend()
    plt.savefig(f'{save_dir}/{save_name}_acc.jpg')
    plt.close()


########################################################################################################################


# 乱数固定関数の定義
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


########################################################################################################################