# syntax=docker/dockerfile:1

# Only TinyTransformer.Api ships - it is the composition root that serves
# both the JSON API and the static frontend (wwwroot) from one Kestrel
# process. TinyTransformer.Core is a dependency, pulled in via ProjectReference.
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Restore first, touching only the files that affect it, so an ordinary code
# change does not invalidate this layer.
COPY Directory.Build.props Directory.Packages.props ./
COPY TinyTransformer.Core/TinyTransformer.Core.csproj TinyTransformer.Core/
COPY TinyTransformer.Api/TinyTransformer.Api.csproj TinyTransformer.Api/
RUN dotnet restore TinyTransformer.Api/TinyTransformer.Api.csproj

COPY TinyTransformer.Core/ TinyTransformer.Core/
COPY TinyTransformer.Api/ TinyTransformer.Api/
RUN dotnet publish TinyTransformer.Api/TinyTransformer.Api.csproj \
    --configuration Release \
    --no-restore \
    --output /app

# Runtime image major version matches the TFM major version (net8.0 -> aspnet:8.0);
# default roll-forward does not cross a major version, so a mismatch here is a
# startup failure, not a warning.
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080
LABEL org.opencontainers.image.source="https://github.com/konradcinkusz/tiny-transformer"
LABEL org.opencontainers.image.description="TinyTransformer: a from-scratch transformer encoder, its API, and its browser demo."

COPY --from=build /app .

# $APP_UID is provided by the base image (a non-root "app" user).
USER $APP_UID
ENTRYPOINT ["dotnet", "TinyTransformer.Api.dll"]
