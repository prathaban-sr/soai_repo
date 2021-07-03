from soai import ModelTrainer

trainer = ModelTrainer()
epoch = 5
trainer.train_test(epoch)
trainer.show_accuracy_loss_graphs()

trainer.show_errors()
trainer.show_attention()
