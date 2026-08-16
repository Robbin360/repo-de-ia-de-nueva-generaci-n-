CREATE TABLE `aethel_chat_messages` (
	`id` int AUTO_INCREMENT NOT NULL,
	`sessionId` varchar(96) NOT NULL,
	`userId` int,
	`role` enum('user','assistant') NOT NULL,
	`content` text NOT NULL,
	`architectureMode` varchar(64) NOT NULL DEFAULT 'hybrid_aethel',
	`tokensProcessed` int NOT NULL DEFAULT 0,
	`metadata` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `aethel_chat_messages_id` PRIMARY KEY(`id`)
);
