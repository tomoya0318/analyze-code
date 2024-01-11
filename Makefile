init:
	@docker compose build --no-cache
	@docker compose up -d

up:
	@docker compose up -d
	
down:
	@docker compose down

exec:
	@docker exec -it 2024_Noguchi bash
