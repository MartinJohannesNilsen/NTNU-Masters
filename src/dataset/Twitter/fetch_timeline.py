import tweepy
from authenticate_twitter import get_auth_api
from dataclasses import dataclass
from datetime import datetime
from typing import List
from tabulate import tabulate

@dataclass
class Simplified_User:
    handle: str
    user_id: int
    name: str

@dataclass
class Simplified_Tweet:
    author: Simplified_User
    created_at: datetime
    full_text: str


def get_timeline(twitter_handle: str, number_of_tweets: int, text_only: bool = False) -> List[Simplified_Tweet]:
    """Get Twitter timeline of posts from user account

    Args:
        twitter_handle (str): Twitter username
        number_of_tweets (int): Amount of tweets in timeline

    Returns:
        [Simplified_Tweet]: List of strings
    """
    try:
        api = get_auth_api()
        user_id = api.get_user(screen_name=twitter_handle).id_str
        timeline = api.user_timeline(user_id=user_id, count=number_of_tweets, tweet_mode="extended")
        if text_only:
            return [tweet.full_text for tweet in timeline]
        return [Simplified_Tweet(author=Simplified_User(handle=tweet.author.screen_name, user_id=tweet.author.id, name=tweet.author.name), created_at=tweet.created_at, full_text=tweet.full_text ) for tweet in timeline]
    except tweepy.errors.NotFound:
        print("Twitter handle not found")
        exit(1)


if __name__ == "__main__":
    posts = get_timeline("elonmusk", 10)
    rows = [[post.created_at.strftime("%d/%m/%y %H:%M:%S"), post.full_text] for post in posts]
    print(tabulate(rows, headers=["Date", "Tweet"]))
