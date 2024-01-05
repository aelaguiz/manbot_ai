aelaguiz@AmirsMacStudio manbot % psql -h 34.170.207.127 -p 5432 -U postgres -W
Password:
psql (14.10 (Homebrew), server 15.4)
WARNING: psql major version 14, server major version 15.
         Some psql features might not work.
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
Type "help" for help.

postgres=> CREATE DATABASE vectordocs;
CREATE DATABASE
postgres=> CREATE EXTENSION vector;
CREATE EXTENSION
postgres=>