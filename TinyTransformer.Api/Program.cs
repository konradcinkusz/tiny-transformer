using System.Threading.RateLimiting;
using TinyTransformer.Api;
using TinyTransformer.Api.Endpoints;
using TinyTransformer.Api.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<EncoderDemoService>();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new()
    {
        Title = "TinyTransformer API",
        Version = "v1",
        Description = "Runs text through a from-scratch transformer encoder block and returns its internals for visualization."
    });
});

// The /api/encode endpoint runs live, unauthenticated matrix math per
// request. There is no user/account system (see docs/architecture/
// DECISIONS.md) to partition by, so this partitions by client IP - the
// documented fallback in architecture-standards' SERVICE-API-PATTERNS.md §1.
builder.Services.AddRateLimiter(options =>
{
    options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    options.AddPolicy(RateLimiterPolicies.Encode, httpContext =>
        RateLimitPartition.GetFixedWindowLimiter(
            partitionKey: httpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown",
            factory: _ => new FixedWindowRateLimiterOptions
            {
                PermitLimit = 30,
                Window = TimeSpan.FromMinutes(1),
                QueueLimit = 0
            }));

    options.OnRejected = async (context, cancellationToken) =>
    {
        double? retryAfterSeconds = context.Lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter)
            ? retryAfter.TotalSeconds
            : null;

        context.HttpContext.Response.ContentType = "application/json";
        await context.HttpContext.Response.WriteAsJsonAsync(
            new { error = "Too many requests. Please slow down.", retryAfter = retryAfterSeconds },
            cancellationToken: cancellationToken);
    };
});

var app = builder.Build();

// Swagger stays on in every environment: this is a public, unauthenticated
// demo API with no secrets behind it, and being explorable is the point.
app.UseSwagger();
app.UseSwaggerUI();

// Frontend and API share one origin (this Kestrel process) - no CORS, no
// token, no proxy layer needed. See docs/architecture/DECISIONS.md for why
// the estate's Next.js/BFF pattern would be disproportionate here.
app.UseDefaultFiles();
app.UseStaticFiles();

app.UseRateLimiter();

app.MapTransformerEndpoints();

app.Run();

// Exposed so TinyTransformer.Api.Tests can boot this app in-process via WebApplicationFactory<Program>.
public partial class Program;
