from soai import ModelTrainer

trainer = ModelTrainer()
start_epoch = 0
train_loss = []
train_acc = []
test_loss = []
test_acc = []

'''
for epoch in range(start_epoch, start_epoch+200):
    train_loss, train_acc = trainer.train(epoch)
    test_loss, test_acc = trainer.test(epoch)
    trainer.scheduler.step()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_accs)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_accs)
axs[1, 1].set_title("Test Accuracy")
'''

trainer.showErrors()
