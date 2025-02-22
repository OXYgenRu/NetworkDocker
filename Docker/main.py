import datetime

from database import DB, Container
from usecase import UseCase

if __name__ == '__main__':
    db = DB()
    use_case = UseCase(db)
    # db.create_tables()
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #
    code = """
    class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMStockPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc3 = nn.Linear(hidden_size, 410)

        self.fc2 = nn.Linear(410, 410)

        self.fc1 = nn.Linear(410, output_size)

        self.tng = nn.LeakyReLU()

        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)

        x = self.fc3(lstm_out[:, -1, :])
        x = self.tng(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.tng(x)
        x = self.drop(x)
        x = self.fc1(x)
        return x
    """
    # opt_code = "torch.optim.Adam(model.parameters(), lr=0.0001)"
    # model = use_case.create_model(False, code)
    # optimizer = use_case.create_optimizer(opt_code)
    # #
    # containers = use_case.create_container(0, model.model_id, optimizer.optimizer_id, True, "test")

    # container = use_case.copy_container(1, True)
    use_case.update_container(2, dataset_id=0, model_id=2, optimizer_id=2, normalise_dataset=True, name="upd_test",
                              comment="")

