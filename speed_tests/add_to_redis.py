from json import JSONEncoder
import itertools
import uuid
import redis
import argparse

encoder = JSONEncoder()


def add_combinations_to_queue(
    models,
    widths,
    heights,
    steps_list,
    outputs,
    redis_connection_string,
    redis_queue_name,
    webhook_url,
    flush_all=False
):
    # Connect to Redis
    r = redis.from_url(redis_connection_string)

    if flush_all:
        r.flushall()
        print('Flushed all keys from Redis')

    # Create the queue if it doesn't exist
    if not r.exists(redis_queue_name):
        r.execute_command(
            'XGROUP', 'CREATE', redis_queue_name,
            redis_queue_name, '$', 'MKSTREAM'
        )
        print(f'Created queue "{redis_queue_name}"')
    else:
        print(f'Queue "{redis_queue_name}" already exists')

    # Create a list of all possible combinations of the input parameters
    combinations = list(itertools.product(
        models, widths, heights, steps_list, outputs
    ))

    # Add a primer to the queue
    model, width, height, steps, output = combinations[0]
    data = get_data(model, width, height, steps, output, webhook_url)
    r.xadd(redis_queue_name, data)
    print(
        f'Primed queue with {model}, {width}, {height}, {steps}, {output}'
    )

    # Iterate through the combinations and add them to the Redis queue
    for model, width, height, steps, output in combinations:
        data = get_data(model, width, height, steps, output, webhook_url)
        r.xadd(redis_queue_name, data)
        print(
            f'Added {model}, {width}, {height}, {steps}, {output} to queue'
        )


def get_data(model, width, height, steps, output, webhook_url):
    input_id = f'test-{str(uuid.uuid4())}'
    val = {
        'webhook_events_filter': ["start", "completed"],
        'webhook': str(webhook_url),
        'input': {
            'id': str(input_id),
            'prompt': 'Portrait of a cat by Van Gogh',
            'width': str(width),
            'height': str(height),
            'inference_steps': str(steps),
            'model': str(model),
            'num_outputs': str(output),
        },
        'response_queue': 'output-queue'
    }
    # convert value to string
    val = encoder.encode(val)
    data = {
        'value': val
    }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+',
                        required=True, help='List of models')
    parser.add_argument('-w', '--widths', nargs='+',
                        required=True, help='List of widths')
    parser.add_argument('-e', '--heights', nargs='+',
                        required=True, help='List of heights')
    parser.add_argument('-s', '--steps-list', nargs='+',
                        required=True, help='List of steps')
    parser.add_argument('-o', '--outputs', nargs='+',
                        required=True, help='List of outputs')
    parser.add_argument('--redis-connection-string',
                        required=True, help='Redis connection string')
    parser.add_argument('--redis-queue-name', required=True,
                        help='Redis queue name')
    parser.add_argument('--webhook-url', required=True, help='Webhook url')
    parser.add_argument('--flush-all', action='store_true')
    args = parser.parse_args()
    add_combinations_to_queue(
        args.models, args.widths, args.heights, args.steps_list, args.outputs,
        args.redis_connection_string, args.redis_queue_name, args.webhook_url,
        args.flush_all
    )
