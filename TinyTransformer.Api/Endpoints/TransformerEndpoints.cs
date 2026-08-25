using TinyTransformer.Api.Contracts;
using TinyTransformer.Api.Services;

namespace TinyTransformer.Api.Endpoints;

// SERVICE-API-PATTERNS.md §2: endpoints stay thin (bind, validate, delegate)
// and are grouped so the route surface is greppable from one place. There is
// only one trust level here - the whole API is unauthenticated by design
// (see docs/architecture/DECISIONS.md) - so there is a single group, not the
// public/auth/admin triad that guide describes for a service with accounts.
public static class TransformerEndpoints
{
    public static IEndpointRouteBuilder MapTransformerEndpoints(this IEndpointRouteBuilder app)
    {
        var api = app.MapGroup("/api");

        api.MapGet("/health", () => Results.Ok(new { status = "healthy" }))
            .WithName("Health")
            .WithSummary("Liveness probe.");

        api.MapPost("/encode", (EncodeRequest request, EncoderDemoService service) =>
            {
                var resolved = ResolvedEncodeRequest.FromRequest(request, Random.Shared.Next());
                var errors = service.Validate(resolved);
                if (errors.Count > 0)
                    return Results.ValidationProblem(errors);

                return Results.Ok(service.Encode(resolved));
            })
            .WithName("EncodeText")
            .WithSummary("Run text through one transformer encoder block and return the embeddings, positional encoding, attention weights, and output for visualization.")
            .RequireRateLimiting(RateLimiterPolicies.Encode);

        return app;
    }
}
