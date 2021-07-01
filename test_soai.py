from soai import ModelTrainer

trainer = ModelTrainer()
start_epoch = 0
for epoch in range(start_epoch, start_epoch+200):
    trainer.train(epoch)
    trainer.test(epoch)
    trainer.scheduler.step()