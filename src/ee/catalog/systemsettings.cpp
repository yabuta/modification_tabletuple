/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */

/* WARNING: THIS FILE IS AUTO-GENERATED
            DO NOT MODIFY THIS SOURCE
            ALL CHANGES MUST BE MADE IN THE CATALOG GENERATOR */

#include <cassert>
#include "systemsettings.h"
#include "catalog.h"

using namespace catalog;
using namespace std;

Systemsettings::Systemsettings(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name)
{
    CatalogValue value;
    m_fields["temptablemaxsize"] = value;
    m_fields["snapshotpriority"] = value;
    m_fields["elasticduration"] = value;
    m_fields["elasticthroughput"] = value;
    m_fields["querytimeout"] = value;
}

Systemsettings::~Systemsettings() {
}

void Systemsettings::update() {
    m_temptablemaxsize = m_fields["temptablemaxsize"].intValue;
    m_snapshotpriority = m_fields["snapshotpriority"].intValue;
    m_elasticduration = m_fields["elasticduration"].intValue;
    m_elasticthroughput = m_fields["elasticthroughput"].intValue;
    m_querytimeout = m_fields["querytimeout"].intValue;
}

CatalogType * Systemsettings::addChild(const std::string &collectionName, const std::string &childName) {
    return NULL;
}

CatalogType * Systemsettings::getChild(const std::string &collectionName, const std::string &childName) const {
    return NULL;
}

bool Systemsettings::removeChild(const std::string &collectionName, const std::string &childName) {
    assert (m_childCollections.find(collectionName) != m_childCollections.end());
    return false;
}

int32_t Systemsettings::temptablemaxsize() const {
    return m_temptablemaxsize;
}

int32_t Systemsettings::snapshotpriority() const {
    return m_snapshotpriority;
}

int32_t Systemsettings::elasticduration() const {
    return m_elasticduration;
}

int32_t Systemsettings::elasticthroughput() const {
    return m_elasticthroughput;
}

int32_t Systemsettings::querytimeout() const {
    return m_querytimeout;
}

