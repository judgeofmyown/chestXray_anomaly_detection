import logging

logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filemode='w'  # 'w' to overwrite every run, 'a' to append
)

for epoch in range(1, 11):
    loss = 0.01 * epoch  # fake loss
    logging.info(f'Epoch {epoch} | Loss: {loss:.4f}')