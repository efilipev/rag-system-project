"""
RabbitMQ Message Queue Service
"""
import json
from typing import Callable, Optional

import aio_pika
from aio_pika import Connection, Channel, Exchange, Queue

from src.core.config import settings
from src.core.logging import logger


class MessageQueueService:
    """
    Service for handling RabbitMQ connections and message publishing/consuming
    """

    def __init__(self):
        self.connection: Optional[Connection] = None
        self.channel: Optional[Channel] = None
        self.exchange: Optional[Exchange] = None

    async def connect(self) -> None:
        """
        Establish connection to RabbitMQ
        """
        try:
            logger.info(f"Connecting to RabbitMQ: {settings.RABBITMQ_URL}")

            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()

            # Set QoS
            await self.channel.set_qos(prefetch_count=settings.MAX_WORKERS)

            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )

            logger.info("Connected to RabbitMQ successfully")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close RabbitMQ connection
        """
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def publish_message(
        self, routing_key: str, message: dict, correlation_id: Optional[str] = None
    ) -> None:
        """
        Publish a message to the exchange
        """
        if not self.exchange:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            # Serialize message
            body = json.dumps(message).encode()

            # Create message
            msg = aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                correlation_id=correlation_id,
            )

            # Publish
            await self.exchange.publish(msg, routing_key=routing_key)

            logger.debug(f"Published message to {routing_key}")

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def consume_messages(
        self, queue_name: str, callback: Callable, routing_key: str = "#"
    ) -> None:
        """
        Start consuming messages from a queue
        """
        if not self.channel or not self.exchange:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            # Declare queue
            queue = await self.channel.declare_queue(queue_name, durable=True)

            # Bind queue to exchange
            await queue.bind(self.exchange, routing_key=routing_key)

            # Start consuming
            await queue.consume(callback)

            logger.info(f"Started consuming from queue: {queue_name}")

        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            raise
