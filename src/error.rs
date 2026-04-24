//! Internal error model for `embed`.

use philharmonic_connector_impl_api::ImplementationError;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum Error {
    #[error("{0}")]
    InvalidConfig(String),

    #[error("{0}")]
    InvalidRequest(String),

    #[error("upstream timeout")]
    UpstreamTimeout,

    #[error("{0}")]
    Internal(String),
}

impl From<Error> for ImplementationError {
    fn from(value: Error) -> Self {
        match value {
            Error::InvalidConfig(detail) => ImplementationError::InvalidConfig { detail },
            Error::InvalidRequest(detail) => ImplementationError::InvalidRequest { detail },
            Error::UpstreamTimeout => ImplementationError::UpstreamTimeout,
            Error::Internal(detail) => ImplementationError::Internal { detail },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_internal_variant_maps_to_wire() {
        let invalid_config = ImplementationError::from(Error::InvalidConfig("cfg".to_owned()));
        assert_eq!(
            invalid_config,
            ImplementationError::InvalidConfig {
                detail: "cfg".to_owned(),
            }
        );

        let invalid_request = ImplementationError::from(Error::InvalidRequest("req".to_owned()));
        assert_eq!(
            invalid_request,
            ImplementationError::InvalidRequest {
                detail: "req".to_owned(),
            }
        );

        let timeout = ImplementationError::from(Error::UpstreamTimeout);
        assert_eq!(timeout, ImplementationError::UpstreamTimeout);

        let internal = ImplementationError::from(Error::Internal("boom".to_owned()));
        assert_eq!(
            internal,
            ImplementationError::Internal {
                detail: "boom".to_owned(),
            }
        );
    }
}
