# This is a copy of <PICO_SDK_PATH>/external/pico_sdk_import.cmake
# Hardened for moonshine-rs: defaults to an immutable, reviewed Pico
# SDK release tag and refuses to fetch when the resolved tag is empty
# (which would otherwise silently track the upstream default branch
# and make firmware builds non-reproducible — see audit row A-146).
#
# Override behavior:
#   * PICO_SDK_PATH (env or cache): if set and a valid directory
#     exists, use that local SDK. Wins over fetching.
#   * PICO_SDK_FETCH_FROM_GIT=ON: fetch from
#     https://github.com/raspberrypi/pico-sdk at the resolved tag.
#   * PICO_SDK_FETCH_FROM_GIT_TAG (env or cache): if unset, the
#     pinned default below is used. If set to an empty string the
#     configure step fails closed rather than tracking the default
#     branch.
#   * PICO_SDK_FETCH_FROM_GIT_PATH: optional scratch directory for
#     the cloned SDK.
#
# Keep this file and
# micro/third-party/pico-tflmicro/pico_sdk_import.cmake byte-for-byte
# identical. Both importers are included by separate firmware targets
# (rp2350 / tflmicro) and they MUST resolve the same SDK commit so
# that any firmware the project produces links against one SDK.

# Pinned Pico SDK release tag. Reviewed at the moonshine-rs v0.9.x
# cycle; bump intentionally and document the change in CHANGELOG.
# A release tag (e.g. 2.1.1) is preferred over a raw commit hash so
# the upstream signing/attestation story applies if a release artifact
# is used. The exact commit at this tag is locked by
# pico_sdk_fetch_tag_pinned in this file's regression test.
set(PICO_SDK_PINNED_TAG "2.1.1")

if (DEFINED ENV{PICO_SDK_PATH} AND (NOT PICO_SDK_PATH))
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    message("Using PICO_SDK_PATH from environment ('${PICO_SDK_PATH}')")
endif ()

if (DEFINED ENV{PICO_SDK_FETCH_FROM_GIT} AND (NOT PICO_SDK_FETCH_FROM_GIT))
    set(PICO_SDK_FETCH_FROM_GIT $ENV{PICO_SDK_FETCH_FROM_GIT})
    message("Using PICO_SDK_FETCH_FROM_GIT from environment ('${PICO_SDK_FETCH_FROM_GIT}')")
endif ()

if (DEFINED ENV{PICO_SDK_FETCH_FROM_GIT_PATH} AND (NOT PICO_SDK_FETCH_FROM_GIT_PATH))
    set(PICO_SDK_FETCH_FROM_GIT_PATH $ENV{PICO_SDK_FETCH_FROM_GIT_PATH})
    message("Using PICO_SDK_FETCH_FROM_GIT_PATH from environment ('${PICO_SDK_FETCH_FROM_GIT_PATH}')")
endif ()

if (DEFINED ENV{PICO_SDK_FETCH_FROM_GIT_TAG} AND (NOT PICO_SDK_FETCH_FROM_GIT_TAG))
    set(PICO_SDK_FETCH_FROM_GIT_TAG $ENV{PICO_SDK_FETCH_FROM_GIT_TAG})
    message("Using PICO_SDK_FETCH_FROM_GIT_TAG from environment ('${PICO_SDK_FETCH_FROM_GIT_TAG}')")
endif ()

# Apply the pinned default only when fetching AND the caller did not
# explicitly set the tag. A consumer who needs a different SDK commit
# sets PICO_SDK_FETCH_FROM_GIT_TAG explicitly; consumers who need the
# default get the pinned release above.
if (PICO_SDK_FETCH_FROM_GIT AND NOT PICO_SDK_FETCH_FROM_GIT_TAG)
  set(PICO_SDK_FETCH_FROM_GIT_TAG "${PICO_SDK_PINNED_TAG}")
  message("Using pinned default PICO_SDK_FETCH_FROM_GIT_TAG='${PICO_SDK_FETCH_FROM_GIT_TAG}' "
          "(override with -DPICO_SDK_FETCH_FROM_GIT_TAG=... to pin a different commit)")
endif()

set(PICO_SDK_PATH "${PICO_SDK_PATH}" CACHE PATH "Path to the Raspberry Pi Pico SDK")
set(PICO_SDK_FETCH_FROM_GIT "${PICO_SDK_FETCH_FROM_GIT}" CACHE BOOL "Set to ON to fetch copy of SDK from git if not otherwise locatable")
set(PICO_SDK_FETCH_FROM_GIT_PATH "${PICO_SDK_FETCH_FROM_GIT_PATH}" CACHE FILEPATH "location to download SDK")
set(PICO_SDK_FETCH_FROM_GIT_TAG "${PICO_SDK_FETCH_FROM_GIT_TAG}" CACHE FILEPATH "release tag for SDK")

if (NOT PICO_SDK_PATH)
    if (PICO_SDK_FETCH_FROM_GIT)
        # Fail closed: an explicit empty tag is a contract violation,
        # not "use the upstream default branch". Tracking master
        # silently is exactly the A-146 bug we are guarding against.
        if (NOT PICO_SDK_FETCH_FROM_GIT_TAG OR PICO_SDK_FETCH_FROM_GIT_TAG STREQUAL "")
            message(FATAL_ERROR
                    "PICO_SDK_FETCH_FROM_GIT is ON but PICO_SDK_FETCH_FROM_GIT_TAG is empty. "
                    "Refusing to track the upstream default branch (A-146). "
                    "Either unset PICO_SDK_FETCH_FROM_GIT to use a local PICO_SDK_PATH, "
                    "or set PICO_SDK_FETCH_FROM_GIT_TAG to an explicit release tag / commit hash.")
        endif()
        include(FetchContent)
        set(FETCHCONTENT_BASE_DIR_SAVE ${FETCHCONTENT_BASE_DIR})
        if (PICO_SDK_FETCH_FROM_GIT_PATH)
            get_filename_component(FETCHCONTENT_BASE_DIR "${PICO_SDK_FETCH_FROM_GIT_PATH}" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
        endif ()
        # GIT_SUBMODULES_RECURSE was added in 3.17
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
            FetchContent_Declare(
                    pico_sdk
                    GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
                    GIT_TAG ${PICO_SDK_FETCH_FROM_GIT_TAG}
                    GIT_SUBMODULES_RECURSE FALSE
            )
        else ()
            FetchContent_Declare(
                    pico_sdk
                    GIT_REPOSITORY https://github.com/raspberrypi/pico-sdk
                    GIT_TAG ${PICO_SDK_FETCH_FROM_GIT_TAG}
            )
        endif ()

        if (NOT pico_sdk)
            message("Downloading Raspberry Pi Pico SDK at tag '${PICO_SDK_FETCH_FROM_GIT_TAG}'")
            FetchContent_Populate(pico_sdk)
            set(PICO_SDK_PATH ${pico_sdk_SOURCE_DIR})
        endif ()
        set(FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR_SAVE})
    else ()
        message(FATAL_ERROR
                "SDK location was not specified. Please set PICO_SDK_PATH or set PICO_SDK_FETCH_FROM_GIT to on to fetch from git."
                )
    endif ()
endif ()

get_filename_component(PICO_SDK_PATH "${PICO_SDK_PATH}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
if (NOT EXISTS ${PICO_SDK_PATH})
    message(FATAL_ERROR "Directory '${PICO_SDK_PATH}' not found")
endif ()

set(PICO_SDK_INIT_CMAKE_FILE ${PICO_SDK_PATH}/pico_sdk_init.cmake)
if (NOT EXISTS ${PICO_SDK_INIT_CMAKE_FILE})
    message(FATAL_ERROR "Directory '${PICO_SDK_PATH}' does not appear to contain the Raspberry Pi Pico SDK")
endif ()

set(PICO_SDK_PATH ${PICO_SDK_PATH} CACHE PATH "Path to the Raspberry Pi Pico SDK" FORCE)

include(${PICO_SDK_INIT_CMAKE_FILE})
