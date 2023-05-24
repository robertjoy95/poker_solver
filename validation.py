from game_classes import Holdem
from player_class import Player, Player_State

def validate_model():
    # test a few basic cases and see what solver does
    game = Holdem(True)
    game.load_nn_model()
    state = Player_State(
                "2hAc",
                [],
                100,
                20,
                0,
                19,
                0,
                6,
                3,
                0.2,
                0.0
            )
    action, amount = game.get_state_action(state)
    print(action)
    print(amount)

if __name__ == "__main__":
    validate_model()