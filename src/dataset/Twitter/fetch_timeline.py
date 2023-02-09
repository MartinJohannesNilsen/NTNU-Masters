import tweepy
from authenticate_twitter import get_auth_api


def get_timeline(twitter_handle: str, number_of_tweets: int) -> list():
    """Get Twitter timeline of posts from user account

    Args:
        twitter_handle (str): Twitter username
        number_of_tweets (int): Amount of tweets in timeline

    Returns:
        [str]: List of strings
    """
    try:
        api = get_auth_api()
        user_id = api.get_user(screen_name=twitter_handle).id_str
        print(user_id)
        timeline = api.user_timeline(user_id=user_id, count=number_of_tweets, tweet_mode="extended")
        # Iterate and print tweets
        textonly_tweets = [tweet.full_text for tweet in timeline]
        return textonly_tweets
    except tweepy.errors.NotFound:
        print("Twitter handle not found")
        exit(1)


if __name__ == "__main__":
    print(*get_timeline("elonmuskdsadsads", 10), sep="\n")
