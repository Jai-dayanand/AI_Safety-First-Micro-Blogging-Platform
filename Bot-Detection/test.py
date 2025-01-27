import pytest
import pandas as pd
import numpy as np
from bot_detection import bot_detection_pipeline

def create_mock_dataset():
    """Creates a mock dataset to test the bot detection pipeline."""
    data = {
        "Tweet": [
            "This is a test tweet!",
            "Another bot-like behavior detected.",
            "I am just a human tweeting",
            "Retweet to win a prize!",
            "Follow me for more updates!",
        ],
        "Retweet Count": [10, 50, 5, 200, 0],
        "Mention Count": [2, 10, 0, 20, 1],
        "Follower Count": [100, 1000, 200, 500, 50],
        "Verified": [False, True, False, True, False],
        "Bot Label": [1, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def mock_dataset(tmp_path_factory):
    """Fixture to create and save a mock dataset to a temporary file."""
    dataset = create_mock_dataset()
    file_path = tmp_path_factory.mktemp("data") / "mock_dataset.csv"
    dataset.to_csv(file_path, index=False)
    return file_path

@pytest.mark.parametrize(
    "input_text, retweet_count, mention_count, follower_count, verified, expected_output",
    [
        ("Retweet to win a prize!", 100, 10, 200, True, 1),  # Clear bot-like behavior
        ("Hello, world!", 0, 0, 100, False, 0),  # Human-like behavior
        ("Breaking news: Something big happened!", 5, 2, 500, True, 0),  # Borderline case
        ("Follow me for free giveaways!", 150, 30, 50, False, 1),  # Likely a bot
        ("This is a normal tweet.", 1, 0, 1000, True, 0),  # Verified user, human
        ("I just love tweeting!", 2, 0, 300, False, 0),  # Normal tweet
        ("Spammy bot behavior detected!", 300, 50, 20, False, 1),  # Bot with high spam metrics
        ("Join us for an event!", 20, 5, 400, True, 0),  # Human-like promotional tweet
        ("Win a free gift now!", 200, 25, 100, False, 1),  # High bot likelihood
        ("Just checking in.", 0, 0, 10, False, 0),  # Very normal human tweet
    ],
)
def test_bot_detection_pipeline(
    mock_dataset,
    input_text,
    retweet_count,
    mention_count,
    follower_count,
    verified,
    expected_output,
):
    """Test the bot detection pipeline on various inputs."""
    result = bot_detection_pipeline(
        dataset_path=str(mock_dataset),
        input_text=input_text,
        retweet_count=retweet_count,
        mention_count=mention_count,
        follower_count=follower_count,
        verified=verified,
    )
    assert result == expected_output, f"Failed for input: {input_text}"

