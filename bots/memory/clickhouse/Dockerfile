# Use the base ClickHouse image
FROM --platform=linux/amd64 clickhouse/clickhouse-server:22.9-alpine

# Set environment variables
ENV ALLOW_EMPTY_PASSWORD=yes
ENV CLICKHOUSE_TCP_PORT=9000
ENV CLICKHOUSE_HTTP_PORT=8123

# Expose ports
EXPOSE 8123 9000

# Set up volumes
VOLUME /var/lib/clickhouse
VOLUME /var/log/clickhouse-server
VOLUME /backups
COPY ./config/backup_disk.xml /etc/clickhouse-server/config.d/backup_disk.xml
COPY ./config/chroma_users.xml /etc/clickhouse-server/users.d/chroma.xml

# Set the entrypoint command
ENTRYPOINT ["clickhouse-server"]