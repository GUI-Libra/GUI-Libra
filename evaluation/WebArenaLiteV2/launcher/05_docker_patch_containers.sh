#!/bin/bash

# stop if any error occur
set -e


# reddit - make server more responsive
docker exec forum sed -i \
  -e 's/^pm.max_children = .*/pm.max_children = 32/' \
  -e 's/^pm.start_servers = .*/pm.start_servers = 10/' \
  -e 's/^pm.min_spare_servers = .*/pm.min_spare_servers = 5/' \
  -e 's/^pm.max_spare_servers = .*/pm.max_spare_servers = 20/' \
  -e 's/^;pm.max_requests = .*/pm.max_requests = 500/' \
  /usr/local/etc/php-fpm.d/www.conf
docker exec forum supervisorctl restart php-fpm

docker exec forum bash -c \
  "sed -i 's|^RATELIMIT_WHITELIST=\$|RATELIMIT_WHITELIST=0.0.0.0/0,::/0|' /var/www/html/.env && \
   sed -i -e 's/@RateLimit(period=\"5 minutes\", max=15,/@RateLimit(period=\"5 minutes\", max=9999,/' \
          -e 's/@RateLimit(period=\"1 hour\", max=3,/@RateLimit(period=\"1 hour\", max=9999,/' \
          /var/www/html/src/DataObject/SubmissionData.php && \
   sed -i 's/@RateLimit(period=\"5 minutes\", max=10,/@RateLimit(period=\"5 minutes\", max=9999,/' \
          /var/www/html/src/DataObject/CommentData.php && \
   rm -rf var/cache/* && php bin/console cache:clear --env=dev -q" || \
  echo "Warning: Failed to disable Reddit rate limit"

# shopping + shopping admin
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$PUBLIC_HOSTNAME:$SHOPPING_PORT" # no trailing /
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$PUBLIC_HOSTNAME:$SHOPPING_PORT/' WHERE path = 'web/secure/base_url';"
# remove the requirement to reset password
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping /var/www/magento2/bin/magento cache:flush

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$PUBLIC_HOSTNAME:$SHOPPING_ADMIN_PORT"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://$PUBLIC_HOSTNAME:$SHOPPING_ADMIN_PORT/' WHERE path = 'web/secure/base_url';"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

# gitlab
docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$PUBLIC_HOSTNAME:$GITLAB_PORT'|" /etc/gitlab/gitlab.rb
docker exec gitlab bash -c "printf '\n\npuma[\"worker_processes\"] = 4' >> /etc/gitlab/gitlab.rb"  # bugfix https://github.com/ServiceNow/BrowserGym/issues/285
docker exec gitlab gitlab-ctl reconfigure

# maps
docker exec openstreetmap-website-web-1 bin/rails db:migrate RAILS_ENV=development

# Import OSM data, only needs to be done once
docker exec openstreetmap-website-web-1 bash -c '
    # Check if database already has data
    NODE_COUNT=$(psql -h db -U openstreetmap -d openstreetmap -t -c "SELECT COUNT(*) FROM nodes LIMIT 1" | tr -d "[:space:]")

    # If node count is 0 or query failed, import data
    if [ "$NODE_COUNT" = "0" ] || [ -z "$NODE_COUNT" ] || [ "$NODE_COUNT" = "f" ]; then
        echo "Database is empty, starting PBF data import..."
        osmosis -verbose \
            --truncate-apidb \
                host="db" \
                database="openstreetmap" \
                user="openstreetmap" \
                validateSchemaVersion="no" \
            --read-pbf /app/pbf_data/pennsylvania-latest.osm.pbf \
            --log-progress \
            --write-apidb \
                host="db" \
                database="openstreetmap" \
                user="openstreetmap" \
                validateSchemaVersion="no"
        echo "PBF data import completed."
    else
        echo "Data already exists in the database, skipping PBF import process."
        echo "Current database has $NODE_COUNT nodes."
    fi
'